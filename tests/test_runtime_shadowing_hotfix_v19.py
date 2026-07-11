from pathlib import Path


def test_pipeline_does_not_shadow_compress_history():
    text = Path("app.py").read_text(encoding="utf-8")

    assert (
        "from context_compressor import "
        "get_subconscious_context, compress_history\n"
    ) not in text
    assert (
        "compress_history as _context_compress_history_v19"
    ) in text
    assert (
        "history = _context_compress_history_v19("
    ) in text


def test_transport_context_guard_is_installed():
    text = Path("groq_client.py").read_text(encoding="utf-8")
    assert "BEGIN CEREBRAS CONTEXT GUARD V19" in text
    assert "_cerebras_fit_context_v19" in text


def test_apo_is_disabled_by_default():
    text = Path("apo_engine.py").read_text(encoding="utf-8")
    assert "BEGIN SAFE APO STARTUP V19" in text
    assert 'ELITE_ENABLE_APO", "0"' in text
