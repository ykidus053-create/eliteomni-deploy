from modules.frontier_provider import estimate_tokens, fit_messages


def test_fit_messages_preserves_system_and_latest_user():
    messages = [
        {"role": "system", "content": "SYSTEM" * 5000},
        {"role": "user", "content": "old" * 10000},
        {"role": "assistant", "content": "answer" * 10000},
        {"role": "user", "content": "LATEST REQUEST" + "x" * 10000},
    ]
    fitted = fit_messages(
        messages,
        context_window=8192,
        output_tokens=2048,
    )
    assert fitted[0]["role"] == "system"
    assert fitted[-1]["role"] == "user"
    assert "LATEST REQUEST" in fitted[-1]["content"]
    assert estimate_tokens(fitted) <= 8192 - 2048 - 512
