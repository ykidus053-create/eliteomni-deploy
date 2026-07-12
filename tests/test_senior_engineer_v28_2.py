from pathlib import Path

import modules.coding_reasoning_v27 as core


ROOT = Path(__file__).resolve().parents[1]


def test_production_is_default():
    contract = core.senior_engineer_contract(
        "Write a database-backed task API."
    )
    assert "MANDATORY SENIOR ENGINEER" in contract
    assert "real production deliverable" in contract
    assert "SQLite" in contract
    assert "transactions" in contract


def test_explicit_tutorial_keeps_senior_quality():
    contract = core.senior_engineer_contract(
        "Write a toy educational parser tutorial."
    )
    assert "reduced teaching or demonstration scope" in contract
    assert "maintainable code" in contract


def test_from_scratch_does_not_enable_toy_scope():
    assert not core.explicit_nonproduction_request(
        "Build a production service from scratch."
    )


def test_verifier_rejects_scope_downgrade(monkeypatch):
    monkeypatch.setattr(
        core,
        "_V282_BASE_VERIFY_CODE_RESPONSE",
        lambda response, language: {
            "approved": True,
            "issues": [],
            "checks": [],
        },
    )

    token = core._V282_ACTIVE_TASK.set(
        "Write a production database service."
    )
    try:
        result = core.verify_code_response(
            """
This is a toy educational implementation.

```python
def run() -> bool:
    return True
```
""",
            "python",
        )
    finally:
        core._V282_ACTIVE_TASK.reset(token)

    assert result["approved"] is False
    assert "scope.toy_implementation" in result["issues"]
    assert "quality.missing_test_or_validation_evidence" in result["issues"]


def test_explicit_tutorial_does_not_trigger_scope_rejection(monkeypatch):
    monkeypatch.setattr(
        core,
        "_V282_BASE_VERIFY_CODE_RESPONSE",
        lambda response, language: {
            "approved": True,
            "issues": [],
            "checks": [],
        },
    )

    token = core._V282_ACTIVE_TASK.set(
        "Write a toy educational parser tutorial."
    )
    try:
        result = core.verify_code_response(
            """
This is a toy educational implementation.

```python
def parse(value: str) -> str:
    return value.strip()

def test_parse() -> None:
    assert parse(" x ") == "x"
```
""",
            "python",
        )
    finally:
        core._V282_ACTIVE_TASK.reset(token)

    assert "scope.toy_implementation" not in result["issues"]


def test_canonical_prompt_is_installed():
    source = (
        ROOT / "modules" / "services" / "prompts.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN SENIOR ENGINEER DEFAULT V28.2" in source
    assert "MANDATORY SENIOR ENGINEER STANDARD" in source


def test_environment_is_documented():
    source = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "ELITE_SENIOR_ENGINEER_DEFAULT=1" in source
    assert "ELITE_PRODUCTION_SCOPE_DEFAULT=1" in source
