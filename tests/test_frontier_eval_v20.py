import importlib.util
from pathlib import Path


def _load():
    path = Path("scripts/run_frontier_eval.py")
    spec = importlib.util.spec_from_file_location("frontier_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_response_checks_sources_code_and_tests():
    module = _load()
    task = {
        "min_chars": 10,
        "must_include": ["rollback"],
        "must_not_include": ["todo"],
        "min_urls": 1,
        "expect_code": True,
        "expect_tests": True,
    }
    response = (
        "rollback\nhttps://example.com\n"
        "```python\ndef test_rollback():\n    assert True\n```"
    )
    result = module.score_response(task, response)
    assert result["score"] == 100.0
