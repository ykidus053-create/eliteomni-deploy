from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_duplicate_stream_rate_limit_is_disabled():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "ELITE_LEGACY_STREAM_RATE_LIMIT" in source
    assert "and not check_rate(ip)" in source


def test_optional_mcp_is_disabled_by_default():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "BEGIN OPTIONAL MCP RUNTIME V33" in source
    assert 'ELITE_ENABLE_MCP", "0"' in source


def test_repetitive_health_logs_are_filterable():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "BEGIN RUNTIME HEALTH QUIET V33" in source
    assert "[MCP STATUS]" in source
    assert "[STATS REPORT]" in source
    assert "[SEARXNG HEALTH]" in source
