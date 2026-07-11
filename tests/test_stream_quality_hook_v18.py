import asyncio
import json

from modules import quality_kernel


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class FakeRoute:
    path = "/stream"

    def __init__(self):
        async def endpoint(request):
            return "legacy"
        self.endpoint = endpoint
        self.app = None


class FakeApp:
    def __init__(self):
        self.routes = [FakeRoute()]


class FakeStreamingResponse:
    def __init__(self, body, **kwargs):
        self.body_iterator = body
        self.kwargs = kwargs


def test_high_risk_stream_is_buffered_through_pipeline(monkeypatch):
    quality_kernel._INSTALLED = False
    app = FakeApp()

    namespace = {
        "app": app,
        "StreamingResponse": FakeStreamingResponse,
        "pipeline_sync": lambda msg, hist: {
            "response": "verified",
            "skill": "coder",
        },
        "classify_skill": lambda msg: "coder",
        "build_system_prompt": lambda *args, **kwargs: "base",
        "build_chatml": lambda system, history, user, *a, **k: [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "FORCE_TOOL_PATTERNS": {},
    }

    quality_kernel.install_runtime_hooks(namespace)
    route = app.routes[0]

    async def collect():
        response = await route.endpoint(FakeRequest({
            "message": "Build a production-grade Python database",
            "history": [],
        }))
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return "".join(chunks)

    output = asyncio.run(collect())
    assert "verified-buffered-v18" in output
    assert output.endswith("verified")
