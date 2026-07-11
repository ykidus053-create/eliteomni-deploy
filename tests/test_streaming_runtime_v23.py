from __future__ import annotations

import asyncio

from modules.streaming_runtime_v23 import (
    AdaptiveTokenStreamMiddleware,
    StreamingMetrics,
    StreamingSettings,
)


def _run(app, path="/stream"):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
    }
    asyncio.run(app(scope, receive, send))
    return sent


def _body(messages):
    return b"".join(
        m.get("body", b"")
        for m in messages
        if m["type"] == "http.response.body"
    )


def _headers(messages):
    start = next(m for m in messages if m["type"] == "http.response.start")
    return {k.lower(): v for k, v in start["headers"]}


def _fake_stream_app(chunks):
    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"transfer-encoding", b"chunked"),
                    (b"content-length", b"999"),
                ],
            }
        )
        for index, chunk in enumerate(chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": index < len(chunks) - 1,
                }
            )

    return app


def test_stream_content_is_byte_for_byte_preserved():
    chunks = [
        b'{"skill":"general","mode":"agentic"}\n',
        b"Hel",
        b"lo ",
        "🌍".encode("utf-8"),
        b"!",
    ]
    settings = StreamingSettings(
        target_bytes=10,
        max_bytes=30,
        max_delay_ms=1000,
        immediate_chunks=1,
    )
    middleware = AdaptiveTokenStreamMiddleware(
        _fake_stream_app(chunks),
        settings=settings,
        metrics=StreamingMetrics(),
    )

    sent = _run(middleware)
    assert _body(sent) == b"".join(chunks)


def test_proxy_buffering_headers_are_repaired():
    settings = StreamingSettings(immediate_chunks=1)
    middleware = AdaptiveTokenStreamMiddleware(
        _fake_stream_app([b"hello"]),
        settings=settings,
        metrics=StreamingMetrics(),
    )

    headers = _headers(_run(middleware))
    assert b"transfer-encoding" not in headers
    assert b"content-length" not in headers
    assert headers[b"x-accel-buffering"] == b"no"
    assert b"no-store" in headers[b"cache-control"]
    assert b"no-transform" in headers[b"cache-control"]


def test_tiny_tokens_are_coalesced_after_immediate_chunks():
    chunks = [
        b'{"skill":"general","mode":"agentic"}\n',
        b"a",
        b"b",
        b"c",
        b"d",
        b"e",
        b"f",
        b"g",
        b"h",
        b"i",
        b"j",
    ]
    settings = StreamingSettings(
        target_bytes=5,
        max_bytes=20,
        max_delay_ms=1000,
        immediate_chunks=1,
        flush_on_newline=False,
        flush_on_punctuation=False,
    )
    metrics = StreamingMetrics()
    middleware = AdaptiveTokenStreamMiddleware(
        _fake_stream_app(chunks),
        settings=settings,
        metrics=metrics,
    )

    sent = _run(middleware)
    body_messages = [
        m for m in sent
        if m["type"] == "http.response.body" and m.get("body")
    ]
    assert _body(sent) == b"".join(chunks)
    assert len(body_messages) < len(chunks)


def test_control_markers_flush_immediately():
    chunks = [
        b'{"skill":"coder","mode":"agentic"}\n',
        b"abc",
        b"\x00THINKING\x00",
        b"reasoning",
        b"\x00/THINKING\x00",
        b"answer",
    ]
    settings = StreamingSettings(
        target_bytes=100,
        max_bytes=200,
        max_delay_ms=1000,
        immediate_chunks=1,
        flush_on_newline=False,
        flush_on_punctuation=False,
    )
    middleware = AdaptiveTokenStreamMiddleware(
        _fake_stream_app(chunks),
        settings=settings,
        metrics=StreamingMetrics(),
    )

    sent = _run(middleware)
    bodies = [
        m.get("body", b"")
        for m in sent
        if m["type"] == "http.response.body"
    ]
    assert b"\x00THINKING\x00" in bodies
    assert b"\x00/THINKING\x00" in bodies
    assert _body(sent) == b"".join(chunks)


def test_non_stream_path_is_not_modified():
    original = _fake_stream_app([b"hello"])
    settings = StreamingSettings(paths=("/stream",))
    middleware = AdaptiveTokenStreamMiddleware(
        original,
        settings=settings,
        metrics=StreamingMetrics(),
    )

    headers = _headers(_run(middleware, path="/chat"))
    assert headers[b"transfer-encoding"] == b"chunked"
    assert headers[b"content-length"] == b"999"


def test_disabled_mode_is_transparent():
    original = _fake_stream_app([b"hello"])
    settings = StreamingSettings(enabled=False)
    middleware = AdaptiveTokenStreamMiddleware(
        original,
        settings=settings,
        metrics=StreamingMetrics(),
    )

    headers = _headers(_run(middleware))
    assert headers[b"transfer-encoding"] == b"chunked"
    assert headers[b"content-length"] == b"999"


def test_settings_are_clamped(monkeypatch):
    monkeypatch.setenv("ELITE_STREAM_TARGET_BYTES", "1")
    monkeypatch.setenv("ELITE_STREAM_MAX_BYTES", "999999")
    monkeypatch.setenv("ELITE_STREAM_MAX_DELAY_MS", "0")
    monkeypatch.setenv("ELITE_STREAM_PATHS", "stream,/api/stream")

    settings = StreamingSettings.from_env()

    assert settings.target_bytes == 8
    assert settings.max_bytes == 4096
    assert settings.max_delay_ms == 5
    assert settings.paths == ("/stream", "/api/stream")
