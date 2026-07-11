from __future__ import annotations

import asyncio
import json

from modules.streaming_runtime_v23 import (
    AdaptiveTokenStreamMiddleware,
    StreamingMetrics,
    StreamingSettings,
)


async def main() -> None:
    settings = StreamingSettings.from_env()
    metrics = StreamingMetrics()
    sent = []

    async def demo_app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (b"content-type", b"text/plain"),
                    (b"transfer-encoding", b"chunked"),
                ],
            }
        )
        pieces = [
            b'{"skill":"general","mode":"agentic"}\n',
            b"Streaming ",
            b"is ",
            b"now ",
            b"smoother ",
            b"on ",
            b"Railway.",
        ]
        for index, piece in enumerate(pieces):
            await send(
                {
                    "type": "http.response.body",
                    "body": piece,
                    "more_body": index < len(pieces) - 1,
                }
            )

    middleware = AdaptiveTokenStreamMiddleware(
        demo_app,
        settings=settings,
        metrics=metrics,
    )

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await middleware(
        {"type": "http", "method": "POST", "path": "/stream", "headers": []},
        receive,
        send,
    )

    headers = next(
        dict(message["headers"])
        for message in sent
        if message["type"] == "http.response.start"
    )
    body_messages = [
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
        and message.get("body")
    ]
    body = b"".join(body_messages).decode("utf-8")

    print(
        json.dumps(
            {
                "version": "V23",
                "settings": {
                    **settings.__dict__,
                    "paths": list(settings.paths),
                },
                "headers": {
                    key.decode(): value.decode()
                    for key, value in headers.items()
                },
                "output_chunks": len(body_messages),
                "body": body,
                "metrics": metrics.snapshot(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
