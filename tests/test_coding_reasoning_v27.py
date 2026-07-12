from __future__ import annotations

from modules.coding_reasoning_v27 import (
    architect_plan,
    detect_language,
    knowledge_boundary_check,
    requirements_contract,
    trim_messages,
    verify_code_response,
)


def test_language_detection_uses_file_reference():
    assert detect_language("Fix src/main.ts line 14") == "typescript"
    assert detect_language("Repair query.sql") == "sql"


def test_message_trim_preserves_system_and_latest_user():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old " * 1000},
        {"role": "assistant", "content": "middle " * 1000},
        {"role": "user", "content": "latest request"},
    ]
    result = trim_messages(messages, max_chars=2200)
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "latest request"
    assert sum(len(item["content"]) for item in result) <= 2200


def test_contract_extracts_traceback_and_language():
    contract = requirements_contract(
        'Fix File "modules/services/agents.py", line 265. '
        "It must preserve the current API."
    )
    assert "python" in contract
    assert "agents.py" in contract
    assert "preserve" in contract


def test_boundary_exists_for_coding():
    result = knowledge_boundary_check(
        "Fix modules/services/agents.py:265",
        "coder",
    )
    assert "CODING KNOWLEDGE BOUNDARY" in result
    assert "agents.py" in result


def test_architect_plan_is_deterministic_and_test_oriented():
    plan = architect_plan("Fix app.py traceback and preserve compatibility")
    assert "focused regression tests" in plan
    assert "full suite" in plan
    assert "smallest complete" in plan


def test_python_verification_accepts_syntax():
    result = verify_code_response(
        "```python\ndef add(a: int, b: int) -> int:\n    return a + b\n```",
        "python",
    )
    assert result["approved"] is True
    assert any("syntax parsed" in item for item in result["checks"])


def test_python_verification_rejects_placeholder():
    result = verify_code_response(
        "```python\ndef add(a, b):\n    # TODO\n    return a + b\n```",
        "python",
    )
    assert result["approved"] is False

