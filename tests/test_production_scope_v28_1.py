from pathlib import Path

from modules.coding_reasoning_v27 import (
    explicit_educational_request,
    production_scope_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ambiguous_database_request_defaults_to_production():
    contract = production_scope_contract("Write a database in Python.")
    assert "PRODUCTION" in contract
    assert "toy" in contract.lower()
    assert "SQLite" in contract or "PostgreSQL" in contract


def test_explicit_tutorial_can_request_educational_mode():
    assert explicit_educational_request(
        "Build a toy educational SQL parser for a tutorial."
    )


def test_from_scratch_alone_does_not_mean_toy():
    assert not explicit_educational_request(
        "Build a production database service from scratch."
    )


def test_canonical_prompts_include_production_scope_guard():
    source = (
        ROOT / "modules" / "services" / "prompts.py"
    ).read_text(encoding="utf-8")
    assert "# BEGIN PRODUCTION SCOPE DEFAULT V28.1" in source
    assert "Do not downgrade" in source


def test_cerebras_reasoning_is_hidden_by_default():
    source = (ROOT / "groq_client.py").read_text(encoding="utf-8")
    assert "# BEGIN HIDDEN REASONING TRANSPORT V28.1" in source
    assert 'ELITE_EXPOSE_MODEL_REASONING", "0"' in source
