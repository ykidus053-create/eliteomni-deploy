from modules.coding_benchmark_v30 import (
    BenchmarkCase,
    evaluate_response,
    load_cases,
    summarize,
)


GOOD_RESPONSE = """
```python
import logging
import os

logger = logging.getLogger(__name__)

def create_user(name: str) -> str:
    try:
        if not name:
            raise ValueError("name is required")
        return name
    except ValueError:
        logger.exception("validation failed")
        raise
```

Use environment variables for configuration. Use parameterized queries,
authentication, authorization, request IDs, metrics, and readiness checks.

```python
def test_create_user() -> None:
    assert create_user("Ada") == "Ada"
```

Validate with: `python3 -m py_compile app.py && pytest -q`
"""


def test_good_response_passes():
    case = BenchmarkCase(
        case_id="good",
        category="api",
        prompt="Build a production API.",
        minimum_score=85,
    )
    result = evaluate_response(case, GOOD_RESPONSE)
    assert result.passed
    assert result.score >= 85
    assert not result.critical_failure


def test_toy_placeholder_response_fails():
    case = BenchmarkCase(
        case_id="bad",
        category="api",
        prompt="Build a production API.",
    )
    result = evaluate_response(
        case,
        "This is a toy implementation. TODO: add tests.",
    )
    assert not result.passed
    assert result.critical_failure


def test_suite_loads_and_summarizes():
    cases = load_cases("benchmarks/coding_v30.jsonl")
    assert len(cases) >= 20
    evaluations = [
        evaluate_response(case, GOOD_RESPONSE)
        for case in cases
    ]
    report = summarize(evaluations)
    assert report["cases"] == len(cases)
    assert "average_score" in report
