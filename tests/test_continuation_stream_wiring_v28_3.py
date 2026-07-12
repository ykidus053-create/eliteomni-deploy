from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_uses_bounded_stream_bridge():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "# BEGIN CONTINUATION STREAMING V28.3" in source
    assert "tok_q = AsyncTokenBridge" in source


def test_legacy_twelve_round_loop_is_removed():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    start = source.index(
        "# BEGIN INTELLIGENT AUTO-CONTINUATION V28.3"
    )
    end = source.index(
        "# END INTELLIGENT AUTO-CONTINUATION V28.3",
        start,
    )
    block = source[start:end]
    assert "_max_continuations = 12" not in block
    assert "should_continue(" in block
    assert "OverlapAwareContinuation" in block


def test_stream_defaults_are_low_latency():
    source = (
        ROOT / "modules" / "streaming_runtime_v23.py"
    ).read_text(encoding="utf-8")
    assert "target_bytes: int = 32" in source
    assert "max_delay_ms: int = 40" in source
    assert "immediate_chunks: int = 4" in source


def test_environment_is_documented():
    source = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ELITE_CONTINUATION_CODER_ROUNDS=3" in source
    assert "ELITE_STREAM_QUEUE_SIZE=128" in source
    assert "ELITE_STREAM_MAX_DELAY_MS=40" in source
