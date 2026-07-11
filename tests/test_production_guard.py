from production_guard import (
    PRODUCTION_CODE_CONTRACT,
    audit_production_response,
    format_audit_for_model,
)


BAD_DATABASE_RESPONSE = r"""
This is a complete production-grade miniature SQL database with ACID,
Write-Ahead Logging, B-Tree O(log N) lookups, and thread-safe networking.

```python
import socket
import threading

DEFAULT_PORT = 5432
WAL_FILE = "mini.wal"

class BTreeNode:
    # Simplified for demo purposes; no splitting logic.
    def insert(self, key, value):
        self.keys.append(key)

class WAL:
    def recover(self):
        try:
            return eval(self.file.read())
        except:
            return []

class Handler(threading.Thread):
    def run(self):
        print("connected")

server.bind(("0.0.0.0", DEFAULT_PORT))
server.listen()
while True:
    conn, addr = server.accept()
    Handler().start()
```
"""


def test_rejects_fake_production_database():
    report = audit_production_response(
        "Build a production-grade mini SQL database",
        BAD_DATABASE_RESPONSE,
    )
    assert report.required is True
    assert report.approved is False
    joined = " ".join(report.violations)
    assert "eval()" in joined
    assert "B-tree" in joined
    assert "unbounded" in joined
    assert "test(s)" in joined


def test_rejects_acid_without_transaction_evidence():
    response = """
    Production-ready ACID storage.
    ```python
    class Store:
        def write(self, value: str) -> None:
            self.value = value
    ```
    """
    report = audit_production_response("Make this production-ready", response)
    assert any("transaction state machine" in v for v in report.violations)


def test_rejects_wal_without_recovery_integrity():
    response = """
    Production durable WAL implementation.
    ```python
    import os
    def append(path: str, data: bytes) -> None:
        with open(path, "ab") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
    ```
    """
    report = audit_production_response("production WAL", response)
    assert any("torn-write" in v for v in report.violations)


def test_non_production_snippet_is_not_forced_through_gate():
    response = """
    ```python
    def add(left: int, right: int) -> int:
        return left + right
    ```
    """
    report = audit_production_response("Show a tiny example", response)
    assert report.required is False
    assert report.approved is True


def test_honest_prototype_is_not_approved_when_production_was_requested():
    response = """
    This is a prototype, not production-ready.
    ```python
    def demo() -> None:
        print("demo")
    ```
    """
    report = audit_production_response("production-ready service", response)
    assert report.approved is False
    assert any("contradicts" in v for v in report.violations)


def test_contract_forbids_unsupported_database_claims():
    lowered = PRODUCTION_CODE_CONTRACT.lower()
    assert "do not claim acid" in lowered
    assert "b-tree must implement node splitting" in lowered
    assert "never deserialize untrusted data with eval" in lowered


def test_failed_report_formats_actionable_feedback():
    report = audit_production_response(
        "production-grade SQL database",
        BAD_DATABASE_RESPONSE,
    )
    formatted = format_audit_for_model(report)
    assert "FAIL" in formatted
    assert "Rewrite the implementation" in formatted


from pathlib import Path

from requirements_matrix import enforce_requirements


def test_requirements_matrix_uses_behavioral_audit():
    response = """
    Production-grade ACID database with a B-tree.
    ```python
    class BTree:
        # simplified; no splitting
        pass
    ```
    """
    result = enforce_requirements(
        response,
        "Build a production-grade SQL database",
    )
    assert result["approved"] is False
    assert result["score"] < 90
    assert result["violations"]


def test_prompt_and_pipeline_are_wired_to_gate():
    root = Path(__file__).resolve().parents[1]
    prompts = (root / "modules/services/prompts.py").read_text(encoding="utf-8")
    pipeline = (root / "modules/services/pipeline.py").read_text(encoding="utf-8")
    root_prompts = (root / "system_prompts.py").read_text(encoding="utf-8")

    assert "BEGIN PRODUCTION EVIDENCE PROMPT V1" in prompts
    assert "BEGIN PRODUCTION EVIDENCE PIPELINE V1" in pipeline
    assert "BEGIN PRODUCTION EVIDENCE ROOT PROMPT V1" in root_prompts
