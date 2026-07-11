from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable


_TRUE = {"1", "true", "yes", "on", "enabled"}
_FALSE = {"0", "false", "no", "off", "disabled"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


@dataclass(frozen=True)
class StreamingSettings:
    enabled: bool = True
    target_bytes: int = 48
    max_bytes: int = 192
    max_delay_ms: int = 70
    immediate_chunks: int = 3
    flush_on_newline: bool = True
    flush_on_punctuation: bool = True
    paths: tuple[str, ...] = ("/stream",)

    @classmethod
    def from_env(cls) -> "StreamingSettings":
        raw_paths = os.getenv("ELITE_STREAM_PATHS", "/stream")
        paths = tuple(
            p.strip() if p.strip().startswith("/") else "/" + p.strip()
            for p in raw_paths.split(",")
            if p.strip()
        ) or ("/stream",)
        return cls(
            enabled=_env_bool("ELITE_STREAM_SMOOTHING", True),
            target_bytes=_env_int("ELITE_STREAM_TARGET_BYTES", 48, 8, 1024),
            max_bytes=_env_int("ELITE_STREAM_MAX_BYTES", 192, 32, 4096),
            max_delay_ms=_env_int("ELITE_STREAM_MAX_DELAY_MS", 70, 5, 1000),
            immediate_chunks=_env_int("ELITE_STREAM_IMMEDIATE_CHUNKS", 3, 0, 20),
            flush_on_newline=_env_bool("ELITE_STREAM_FLUSH_NEWLINE", True),
            flush_on_punctuation=_env_bool("ELITE_STREAM_FLUSH_PUNCTUATION", True),
            paths=paths,
        )


class StreamingMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.input_chunks = 0
        self.output_chunks = 0
        self.input_bytes = 0
        self.output_bytes = 0
        self.last_ttft_ms: float | None = None
        self.last_duration_ms: float | None = None

    def start(self) -> float:
        with self._lock:
            self.requests += 1
        return time.perf_counter()

    def input(self, size: int) -> None:
        with self._lock:
            self.input_chunks += 1
            self.input_bytes += size

    def output(self, size: int, started: float, first: bool = False) -> None:
        now = time.perf_counter()
        with self._lock:
            self.output_chunks += 1
            self.output_bytes += size
            if first:
                self.last_ttft_ms = round((now - started) * 1000, 2)

    def finish(self, started: float) -> None:
        with self._lock:
            self.last_duration_ms = round((time.perf_counter() - started) * 1000, 2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ratio = (
                round(self.output_chunks / self.input_chunks, 3)
                if self.input_chunks
                else None
            )
            return {
                "requests": self.requests,
                "input_chunks": self.input_chunks,
                "output_chunks": self.output_chunks,
                "input_bytes": self.input_bytes,
                "output_bytes": self.output_bytes,
                "chunk_ratio": ratio,
                "last_ttft_ms": self.last_ttft_ms,
                "last_duration_ms": self.last_duration_ms,
            }


def _header_name(raw: bytes) -> bytes:
    return raw.strip().lower()


def _set_header(
    headers: list[tuple[bytes, bytes]],
    name: bytes,
    value: bytes,
) -> None:
    target = _header_name(name)
    headers[:] = [(k, v) for k, v in headers if _header_name(k) != target]
    headers.append((name, value))


def _remove_headers(
    headers: list[tuple[bytes, bytes]],
    names: Iterable[bytes],
) -> None:
    blocked = {_header_name(name) for name in names}
    headers[:] = [
        (k, v) for k, v in headers if _header_name(k) not in blocked
    ]


def _looks_like_metadata(body: bytes) -> bool:
    stripped = body.strip()
    return (
        stripped.startswith(b"{")
        and stripped.endswith(b"}")
        and b"\n" in body
        and b'"skill"' in stripped
    )


_CONTROL_MARKERS = (
    b"\x00THINKING\x00",
    b"\x00/THINKING\x00",
    b"\x00FINISH_REASON\x00",
    b"\x00TOOLCALL\x00",
)


def _is_control_chunk(body: bytes) -> bool:
    return _looks_like_metadata(body) or any(
        marker in body for marker in _CONTROL_MARKERS
    )


def _ends_at_natural_boundary(body: bytearray, settings: StreamingSettings) -> bool:
    if not body:
        return False
    if settings.flush_on_newline and (body.endswith(b"\n") or body.endswith(b"\r")):
        return True
    if settings.flush_on_punctuation:
        stripped = bytes(body).rstrip()
        return stripped.endswith((b".", b"!", b"?", b":", b";", b","))
    return False


class AdaptiveTokenStreamMiddleware:
    """
    ASGI middleware for EliteOmni's existing raw text streaming protocol.

    It preserves control/meta chunks, removes proxy-hostile framing headers,
    marks the response non-cacheable, and coalesces tiny provider fragments
    into smoother browser-sized chunks without changing response content.
    """

    def __init__(
        self,
        app: Any,
        settings: StreamingSettings | None = None,
        metrics: StreamingMetrics | None = None,
    ) -> None:
        self.app = app
        self.settings = settings or StreamingSettings.from_env()
        self.metrics = metrics or StreamingMetrics()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            not self.settings.enabled
            or scope.get("type") != "http"
            or scope.get("path") not in self.settings.paths
        ):
            await self.app(scope, receive, send)
            return

        pending = bytearray()
        message_count = 0
        emitted_count = 0
        started = self.metrics.start()
        last_emit = time.perf_counter()
        finished = False

        async def emit(body: bytes, more_body: bool) -> None:
            nonlocal emitted_count, last_emit
            if body:
                emitted_count += 1
                self.metrics.output(
                    len(body),
                    started,
                    first=(emitted_count == 1),
                )
            await send(
                {
                    "type": "http.response.body",
                    "body": body,
                    "more_body": more_body,
                }
            )
            last_emit = time.perf_counter()

        async def flush_pending(more_body: bool) -> None:
            if not pending:
                if not more_body:
                    await emit(b"", False)
                return
            body = bytes(pending)
            pending.clear()
            await emit(body, more_body)

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal message_count, last_emit, finished

            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                _remove_headers(
                    headers,
                    (
                        b"content-length",
                        b"transfer-encoding",
                    ),
                )
                _set_header(
                    headers,
                    b"cache-control",
                    b"no-cache, no-store, must-revalidate, no-transform",
                )
                _set_header(headers, b"pragma", b"no-cache")
                _set_header(headers, b"expires", b"0")
                _set_header(headers, b"x-accel-buffering", b"no")
                _set_header(headers, b"x-content-type-options", b"nosniff")
                message = dict(message)
                message["headers"] = headers
                await send(message)
                return

            if message["type"] != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"") or b""
            if isinstance(body, str):
                body = body.encode("utf-8")
            more_body = bool(message.get("more_body", False))

            if body:
                message_count += 1
                self.metrics.input(len(body))

            immediate = bool(body) and (
                message_count <= self.settings.immediate_chunks
                or _is_control_chunk(body)
            )

            if immediate:
                if pending:
                    await flush_pending(True)
                await emit(body, more_body)
                if not more_body:
                    finished = True
                    self.metrics.finish(started)
                return

            if body:
                pending.extend(body)

            elapsed_ms = (time.perf_counter() - last_emit) * 1000
            should_flush = bool(pending) and (
                len(pending) >= self.settings.target_bytes
                or len(pending) >= self.settings.max_bytes
                or elapsed_ms >= self.settings.max_delay_ms
                or _ends_at_natural_boundary(pending, self.settings)
                or not more_body
            )

            if should_flush:
                await flush_pending(more_body)
            elif not more_body:
                await flush_pending(False)

            if not more_body:
                finished = True
                self.metrics.finish(started)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if not finished:
                self.metrics.finish(started)


def install_streaming_runtime_v23(app: Any) -> dict[str, Any]:
    if getattr(app.state, "streaming_runtime_v23_installed", False):
        return {
            "installed": True,
            "already_installed": True,
        }

    settings = StreamingSettings.from_env()
    metrics = StreamingMetrics()

    app.add_middleware(
        AdaptiveTokenStreamMiddleware,
        settings=settings,
        metrics=metrics,
    )

    async def streaming_status_v23() -> dict[str, Any]:
        return {
            "version": "V23",
            "installed": True,
            "settings": {
                **asdict(settings),
                "paths": list(settings.paths),
            },
            "metrics": metrics.snapshot(),
        }

    existing_paths = {
        getattr(route, "path", None)
        for route in getattr(app, "routes", [])
    }
    if "/stream/status" not in existing_paths:
        app.add_api_route(
            "/stream/status",
            streaming_status_v23,
            methods=["GET"],
            name="streaming_status_v23",
        )

    app.state.streaming_runtime_v23_installed = True
    app.state.streaming_runtime_v23_settings = settings
    app.state.streaming_runtime_v23_metrics = metrics

    return {
        "installed": True,
        "already_installed": False,
        "settings": {
            **asdict(settings),
            "paths": list(settings.paths),
        },
    }


__all__ = [
    "AdaptiveTokenStreamMiddleware",
    "StreamingMetrics",
    "StreamingSettings",
    "install_streaming_runtime_v23",
]
