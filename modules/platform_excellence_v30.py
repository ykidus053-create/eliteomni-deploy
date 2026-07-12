"""Production runtime hardening and telemetry for EliteOmni V30."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Awaitable, Callable, Mapping

from starlette.responses import JSONResponse

log = logging.getLogger("eliteomni.platform")


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _env_float(
    name: str,
    default: float,
    low: float,
    high: float,
) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(
        item.strip()
        for item in raw.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class RuntimeSettings:
    allowed_origins: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    max_request_bytes: int
    max_concurrency: int
    concurrency_wait_seconds: float
    requests_per_minute: int
    rate_limit_burst: int
    required_env: tuple[str, ...]
    enable_metrics: bool
    trust_proxy_headers: bool

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        return cls(
            allowed_origins=_csv_env(
                "ELITE_ALLOWED_ORIGINS",
                "http://localhost:3000,http://127.0.0.1:3000",
            ),
            allowed_hosts=_csv_env("ELITE_ALLOWED_HOSTS"),
            max_request_bytes=_env_int(
                "ELITE_MAX_REQUEST_BYTES",
                10 * 1024 * 1024,
                1024,
                100 * 1024 * 1024,
            ),
            max_concurrency=_env_int(
                "ELITE_MAX_CONCURRENT_REQUESTS",
                32,
                1,
                512,
            ),
            concurrency_wait_seconds=_env_float(
                "ELITE_CONCURRENCY_WAIT_SECONDS",
                2.0,
                0.05,
                30.0,
            ),
            requests_per_minute=_env_int(
                "ELITE_REQUESTS_PER_MINUTE",
                120,
                1,
                10000,
            ),
            rate_limit_burst=_env_int(
                "ELITE_RATE_LIMIT_BURST",
                30,
                1,
                1000,
            ),
            required_env=_csv_env(
                "ELITE_REQUIRED_ENV",
                "CEREBRAS_API_KEY",
            ),
            enable_metrics=(
                os.getenv("ELITE_ENABLE_INTERNAL_METRICS", "1") == "1"
            ),
            trust_proxy_headers=(
                os.getenv("ELITE_TRUST_PROXY_HEADERS", "1") == "1"
            ),
        )


class RuntimeMetrics:
    """Small process-local metrics registry without external dependencies."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._latencies_ms: deque[float] = deque(maxlen=5000)
        self._inflight = 0

    def begin(self) -> None:
        with self._lock:
            self._counters["requests_total"] += 1
            self._inflight += 1
            self._counters["inflight_peak"] = max(
                self._counters["inflight_peak"],
                self._inflight,
            )

    def finish(self, status: int, elapsed_ms: float) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._counters[f"status_{status}"] += 1
            if status >= 500:
                self._counters["server_errors_total"] += 1
            elif status >= 400:
                self._counters["client_errors_total"] += 1
            self._latencies_ms.append(elapsed_ms)

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] += value

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ordered = sorted(self._latencies_ms)

            def percentile(p: float) -> float:
                if not ordered:
                    return 0.0
                index = min(
                    len(ordered) - 1,
                    max(0, int((len(ordered) - 1) * p)),
                )
                return round(ordered[index], 2)

            return {
                "version": "V30",
                "uptime_seconds": round(time.time() - self._started, 2),
                "inflight": self._inflight,
                "counters": dict(self._counters),
                "latency_ms": {
                    "p50": percentile(0.50),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "samples": len(ordered),
                },
            }


METRICS = RuntimeMetrics()


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    def __init__(self, per_minute: int, burst: int) -> None:
        self.rate_per_second = per_minute / 60.0
        self.capacity = float(max(1, burst))
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    tokens=self.capacity,
                    updated_at=current,
                )
                self._buckets[key] = bucket

            elapsed = max(0.0, current - bucket.updated_at)
            bucket.tokens = min(
                self.capacity,
                bucket.tokens + elapsed * self.rate_per_second,
            )
            bucket.updated_at = current

            if bucket.tokens < 1.0:
                return False
            bucket.tokens -= 1.0

            if len(self._buckets) > 10000:
                cutoff = current - 3600
                self._buckets = {
                    item_key: item
                    for item_key, item in self._buckets.items()
                    if item.updated_at >= cutoff
                }
            return True


def _headers(scope: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in scope.get("headers", []):
        result[key.decode("latin-1").lower()] = value.decode("latin-1")
    return result


def _client_key(scope: Mapping[str, Any], settings: RuntimeSettings) -> str:
    headers = _headers(scope)
    if settings.trust_proxy_headers:
        forwarded = headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",", 1)[0].strip()
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def admin_token_valid(request: Any) -> bool:
    """Validate an admin token with no insecure fallback value."""
    expected = os.getenv("ELITE_ADMIN_TOKEN", "")
    if len(expected) < 24:
        return False

    supplied = ""
    headers = getattr(request, "headers", None)
    if headers is not None:
        supplied = headers.get("x-elite-admin-token", "")
        if not supplied:
            authorization = headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                supplied = authorization[7:].strip()

    if (
        not supplied
        and os.getenv("ELITE_ALLOW_ADMIN_QUERY_TOKEN", "0") == "1"
    ):
        query = getattr(request, "query_params", None)
        if query is not None:
            supplied = query.get("secret", "")

    return bool(supplied) and hmac.compare_digest(
        supplied.encode("utf-8"),
        expected.encode("utf-8"),
    )


def readiness_report(
    settings: RuntimeSettings | None = None,
) -> dict[str, Any]:
    config = settings or RuntimeSettings.from_env()
    missing = [
        name
        for name in config.required_env
        if not os.getenv(name, "").strip()
    ]
    return {
        "ok": not missing,
        "version": "V30",
        "missing_required_environment": missing,
        "runtime": {
            "max_request_bytes": config.max_request_bytes,
            "max_concurrency": config.max_concurrency,
            "requests_per_minute": config.requests_per_minute,
            "allowed_origins_count": len(config.allowed_origins),
        },
    }


class PlatformGuardMiddleware:
    """Pure ASGI middleware that preserves streaming responses."""

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        settings: RuntimeSettings,
    ) -> None:
        self.app = app
        self.settings = settings
        self._semaphore = asyncio.Semaphore(
            settings.max_concurrency
        )
        self._limiter = TokenBucketLimiter(
            settings.requests_per_minute,
            settings.rate_limit_burst,
        )

    async def _reject(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[None]],
        *,
        status: int,
        message: str,
        request_id: str,
    ) -> None:
        response = JSONResponse(
            {
                "error": message,
                "request_id": request_id,
            },
            status_code=status,
            headers={
                "x-request-id": request_id,
                "cache-control": "no-store",
            },
        )
        await response(scope, receive, send)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Awaitable[Any]],
        send: Callable[..., Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = _headers(scope)
        supplied_request_id = headers.get("x-request-id", "")
        request_id = (
            supplied_request_id[:128]
            if supplied_request_id.isascii()
            else ""
        ) or uuid.uuid4().hex
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "GET")).upper()

        content_length = headers.get("content-length", "")
        if content_length:
            try:
                too_large = (
                    int(content_length)
                    > self.settings.max_request_bytes
                )
            except ValueError:
                too_large = True
            if too_large:
                METRICS.increment("request_too_large_total")
                await self._reject(
                    scope,
                    receive,
                    send,
                    status=413,
                    message="request body exceeds configured limit",
                    request_id=request_id,
                )
                return

        if path not in {
            "/api/health/live",
            "/api/health/ready",
        }:
            key = _client_key(scope, self.settings)
            if not self._limiter.allow(key):
                METRICS.increment("rate_limited_total")
                await self._reject(
                    scope,
                    receive,
                    send,
                    status=429,
                    message="rate limit exceeded",
                    request_id=request_id,
                )
                return

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.settings.concurrency_wait_seconds,
            )
        except asyncio.TimeoutError:
            METRICS.increment("concurrency_rejected_total")
            await self._reject(
                scope,
                receive,
                send,
                status=503,
                message="server is at its concurrency limit",
                request_id=request_id,
            )
            return

        started = time.perf_counter()
        status_code = 500
        METRICS.begin()

        async def send_with_headers(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    [
                        (b"x-request-id", request_id.encode("ascii")),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (
                            b"referrer-policy",
                            b"strict-origin-when-cross-origin",
                        ),
                        (
                            b"permissions-policy",
                            b"geolocation=(), payment=(), usb=()",
                        ),
                        (
                            b"content-security-policy",
                            (
                                b"default-src 'self' https: data: blob:; "
                                b"script-src 'self' 'unsafe-inline' https:; "
                                b"style-src 'self' 'unsafe-inline' https:; "
                                b"img-src 'self' data: blob: https:; "
                                b"connect-src 'self' https: wss:; "
                                b"frame-ancestors 'none'"
                            ),
                        ),
                    ]
                )
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except asyncio.CancelledError:
            status_code = 499
            METRICS.increment("client_disconnect_total")
            raise
        except Exception:
            status_code = 500
            METRICS.increment("unhandled_exception_total")
            raise
        finally:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0
            METRICS.finish(status_code, elapsed_ms)
            self._semaphore.release()
            log.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": method,
                        "path": path,
                        "status": status_code,
                        "latency_ms": round(elapsed_ms, 2),
                    },
                    separators=(",", ":"),
                )
            )


def _route_exists(app: Any, path: str) -> bool:
    return any(
        getattr(route, "path", None) == path
        for route in getattr(app, "routes", [])
    )


def configure_platform_excellence(app: Any) -> RuntimeSettings:
    """Install hardened middleware and evidence-oriented endpoints."""
    from fastapi import Request
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    settings = RuntimeSettings.from_env()

    app.add_middleware(
        PlatformGuardMiddleware,
        settings=settings,
    )

    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=[
                "authorization",
                "content-type",
                "x-request-id",
                "x-elite-admin-token",
            ],
            expose_headers=["x-request-id"],
            max_age=600,
        )

    if settings.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=list(settings.allowed_hosts),
        )

    if not _route_exists(app, "/api/health/live"):
        async def live() -> dict[str, Any]:
            return {
                "ok": True,
                "version": "V30",
                "time": int(time.time()),
            }

        app.add_api_route(
            "/api/health/live",
            live,
            methods=["GET"],
            include_in_schema=False,
        )

    if not _route_exists(app, "/api/health/ready"):
        async def ready() -> JSONResponse:
            report = readiness_report(settings)
            return JSONResponse(
                report,
                status_code=200 if report["ok"] else 503,
            )

        app.add_api_route(
            "/api/health/ready",
            ready,
            methods=["GET"],
            include_in_schema=False,
        )

    if (
        settings.enable_metrics
        and not _route_exists(app, "/internal/metrics")
    ):
        async def metrics(request: Request) -> JSONResponse:
            if not admin_token_valid(request):
                return JSONResponse(
                    {"error": "unauthorized"},
                    status_code=403,
                )
            return JSONResponse(METRICS.snapshot())

        app.add_api_route(
            "/internal/metrics",
            metrics,
            methods=["GET"],
            include_in_schema=False,
        )

    return settings


def runtime_status() -> dict[str, Any]:
    settings = RuntimeSettings.from_env()
    return {
        "version": "V30",
        "settings": asdict(settings),
        "readiness": readiness_report(settings),
        "metrics": METRICS.snapshot(),
    }


__all__ = [
    "METRICS",
    "PlatformGuardMiddleware",
    "RuntimeMetrics",
    "RuntimeSettings",
    "TokenBucketLimiter",
    "admin_token_valid",
    "configure_platform_excellence",
    "readiness_report",
    "runtime_status",
]
