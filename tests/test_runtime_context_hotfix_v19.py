import os

import groq_client


def test_cerebras_context_guard_fits_8192_window(monkeypatch):
    monkeypatch.setenv("CEREBRAS_CONTEXT_WINDOW", "8192")
    monkeypatch.setenv("CEREBRAS_MAX_OUTPUT_TOKENS", "3072")

    messages = [
        {"role": "system", "content": "S" * 30000},
        {"role": "user", "content": "old " * 5000},
        {"role": "assistant", "content": "answer " * 5000},
        {"role": "user", "content": "LATEST QUESTION " + ("x" * 12000)},
    ]

    fitted, output_tokens = groq_client._cerebras_fit_context_v19(
        messages,
        16000,
    )

    assert output_tokens == 3072
    assert groq_client._cerebras_estimate_tokens_v19(fitted) <= (
        8192 - output_tokens - 384
    )
    assert fitted[-1]["role"] == "user"
    assert "LATEST QUESTION" in fitted[-1]["content"]


def test_cerebras_context_guard_keeps_system_and_latest_user():
    fitted, _ = groq_client._cerebras_fit_context_v19(
        [
            {"role": "system", "content": "SYSTEM CONTRACT"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "LATEST"},
        ],
        1000,
    )

    assert fitted[0]["role"] == "system"
    assert "SYSTEM CONTRACT" in fitted[0]["content"]
    assert fitted[-1]["role"] == "user"
    assert fitted[-1]["content"] == "LATEST"
