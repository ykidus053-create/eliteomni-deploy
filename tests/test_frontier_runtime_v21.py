import os

from modules.frontier_runtime_v21 import (
    _requirements_contract,
    install_frontier_runtime_v21,
)


def test_requirements_contract_preserves_explicit_constraints():
    contract = _requirements_contract(
        "Fix app.py:42. Must include tests. Do not change the API."
    )
    assert "Must include tests" in contract
    assert "Do not change the API" in contract
    assert "app.py" in contract


def test_v21_runs_independent_candidate_for_production_task(monkeypatch):
    monkeypatch.setenv("ELITE_FRONTIER_V21_MODE", "balanced")
    monkeypatch.setenv("ELITE_FRONTIER_V21_CANDIDATES", "2")
    calls = []

    def fake_pipeline(message, history):
        calls.append(message)
        if len(calls) == 1:
            return {
                "response": "I withheld the generated answer because verification failed.",
                "quality_v18": {"approved": False},
            }
        return {
            "response": (
                "```python\n"
                "def add(a, b):\n    return a + b\n\n"
                "def test_add():\n    assert add(2, 3) == 5\n"
                "```"
            ),
            "quality_v18": {"approved": True},
        }

    namespace = {
        "pipeline_sync": fake_pipeline,
        "classify_skill": lambda _: "coder",
    }

    import modules.frontier_runtime_v21 as runtime
    monkeypatch.setattr(runtime, "_INSTALLED", False)
    install_frontier_runtime_v21(namespace)

    result = namespace["pipeline_sync"](
        "Build production-grade Python code with executable tests.",
        [],
    )

    assert len(calls) == 2
    assert result["frontier_v21"]["candidate_count"] == 2
    assert result["frontier_v21"]["selected"] in {
        "baseline",
        "independent-challenger",
    }
