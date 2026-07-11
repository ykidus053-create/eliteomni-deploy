from modules.quality_kernel import (
    analyze_request,
    audit_answer,
    compact_system_prompt,
    needs_fresh_research,
)


def test_general_explanation_does_not_force_search():
    assert needs_fresh_research("Explain how a B-tree works") is False


def test_current_release_forces_search():
    assert needs_fresh_research(
        "What is the latest PostgreSQL release today?"
    ) is True


def test_system_prompt_is_deduplicated_and_bounded():
    profile = analyze_request("Build a Python API")
    repeated = ("Same paragraph.\n\n" * 5000) + "LIVE END"
    compacted = compact_system_prompt(repeated, profile, max_chars=9000)
    assert len(compacted) <= 9000
    assert compacted.count("Same paragraph.") == 1
    assert "CODING CONTRACT" in compacted


def test_bad_database_answer_is_rejected():
    response = r"""
A complete production-grade ACID database with two-phase commit.

```python
import socket
import threading

class Database:
    def __init__(self):
        self.current_tx_id = None
        self.tx_buffer = []

    def insert(self, row):
        self.pk_index.insert(row["id"], row)

    def rollback(self):
        self.tx_buffer = []

class Handler(socket.socket):
    def handle(self, client):
        query = client.recv(4096)

threading.Thread(target=lambda: None).start()
```
"""
    audit = audit_answer(
        "Build a production-grade miniature SQL database in Python",
        response,
    )
    codes = {issue.code for issue in audit.issues}
    assert audit.approved is False
    assert "db.shared_transaction_state" in codes
    assert "db.rollback_does_not_undo" in codes
    assert "db.tcp_without_framing" in codes


def test_current_research_without_sources_is_rejected():
    audit = audit_answer(
        "Research the latest database developments today",
        "Several major changes happened.",
    )
    assert audit.approved is False
    assert any(
        issue.code == "research.missing_sources"
        for issue in audit.issues
    )
