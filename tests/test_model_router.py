import time

import model_router as router


def setup_function():
    router.CircuitState._state.clear()


def test_system_prompt_code_words_do_not_force_coder():
    payload = [
        {"role": "system", "content": "You are a Python-aware assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    assert router.select_model("easy", payload) == router.MISTRAL_SMALL


def test_latest_user_code_request_routes_to_coder():
    payload = [
        {"role": "user", "content": "Write a Python function to sort records."}
    ]
    assert router.select_model("medium", payload) == router.CODESTRAL


def test_hard_request_routes_to_large_model():
    payload = [
        {"role": "user", "content": "Analyze these competing strategies."}
    ]
    assert router.select_model("hard", payload) == router.MISTRAL_LARGE


def test_trim_system_keeps_opening_and_closing_context():
    prompt = "POLICY-START\n" + ("x" * 500) + "\nLATEST-CONTEXT-END"
    trimmed = router.trim_system(prompt, max_tokens=30)
    assert trimmed.startswith("POLICY-START")
    assert trimmed.endswith("LATEST-CONTEXT-END")
    assert "system prompt trimmed" in trimmed
    assert len(trimmed) <= 120


def test_circuit_opens_and_resets(monkeypatch):
    monkeypatch.setattr(router.CircuitState, "THRESHOLD", 2)
    monkeypatch.setattr(router.CircuitState, "RESET_S", 1)

    router.CircuitState.record_failure("model-a")
    assert router.CircuitState.is_open("model-a") is False

    router.CircuitState.record_failure("model-a")
    assert router.CircuitState.is_open("model-a") is True

    router.CircuitState._state["model-a"]["opened_at"] = time.monotonic() - 2
    assert router.CircuitState.is_open("model-a") is False


def test_fallback_skips_open_model(monkeypatch):
    monkeypatch.setattr(router, "MISTRAL_LARGE", "model-large")
    monkeypatch.setattr(router, "MISTRAL_MEDIUM", "model-medium")
    monkeypatch.setattr(router, "MISTRAL_SMALL", "model-small")
    monkeypatch.setattr(router, "CODESTRAL", "model-code")
    monkeypatch.setattr(router, "MAGISTRAL", "model-reasoning")
    monkeypatch.setattr(
        router,
        "FALLBACK_CHAIN",
        {"model-large": "model-medium"},
    )
    monkeypatch.setattr(router.CircuitState, "THRESHOLD", 1)

    router.CircuitState.record_failure("model-large")
    assert router.route_with_fallback("model-large") == "model-medium"
