from pathlib import Path

import modules.services.tool_schemas as tool_schemas


ROOT = Path(__file__).resolve().parents[1]


def test_v2_and_v31_stream_markers_coexist():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "# BEGIN STREAM PRODUCTION ROUTE V2" in source
    assert "# END STREAM PRODUCTION ROUTE V2" in source
    assert "# BEGIN STREAM CODER VERIFIED ROUTE V31" in source
    assert "# END STREAM CODER VERIFIED ROUTE V31" in source


def test_fetch_empty_result_becomes_truthful_status(monkeypatch):
    monkeypatch.setattr(
        tool_schemas,
        "_dispatch_tool_call_without_offline_result",
        lambda name, args: "",
    )
    result = tool_schemas.dispatch_tool_call(
        "fetch",
        {"url": "https://example.invalid"},
    )
    assert "Fetch unavailable" in result
    assert "fabricated" in result


def test_nonempty_fetch_result_is_preserved(monkeypatch):
    monkeypatch.setattr(
        tool_schemas,
        "_dispatch_tool_call_without_offline_result",
        lambda name, args: "page content",
    )
    assert (
        tool_schemas.dispatch_tool_call(
            "fetch",
            {"url": "https://example.test"},
        )
        == "page content"
    )
