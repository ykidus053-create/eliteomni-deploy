from pathlib import Path

from production_guard import audit_production_response


BAD_DATABASE_RESPONSE = r"""
This is a complete production-grade ACID miniature SQL database using
two-phase commit, WAL durability, and thread-safe transactions.

```python
import json
import socket
import threading

class Database:
    def __init__(self):
        self.current_tx_id = None
        self.tx_buffer = []

    def begin(self):
        self.current_tx_id = 1
        self.tx_buffer = []

    def insert(self, table, row):
        self.pk_index.insert(row["id"], row)
        if self.current_tx_id is None:
            self.storage.persist_row(table, row)

    def rollback(self):
        self.tx_buffer = []
        self.current_tx_id = None

class DatabaseRequestHandler(socket.socket):
    def handle(self, client_sock):
        query = client_sock.recv(4096).decode()

def worker(db):
    db.begin()

threading.Thread(target=worker, args=(Database(),)).start()
```
"""


def test_bad_shared_transaction_database_is_rejected():
    report = audit_production_response(
        "Build a production-grade miniature SQL database",
        BAD_DATABASE_RESPONSE,
    )

    assert report.approved is False
    joined = "\n".join(report.violations).lower()
    assert "transaction" in joined and "shared" in joined
    assert "rollback" in joined or "undo" in joined
    assert "tcp" in joined or "framing" in joined
    assert "socket.socket" in joined


def test_unclosed_code_fence_is_rejected():
    report = audit_production_response(
        "Build a production-grade database",
        "Production-grade database.\n```python\nclass Database:\n    pass\n",
    )
    assert report.approved is False
    assert any(
        "unclosed code fence" in item
        for item in report.violations
    )


def test_actual_output_paths_use_final_gate():
    root = Path(__file__).resolve().parents[1]
    app = (root / "app.py").read_text(encoding="utf-8")
    pipeline = (
        root / "modules/services/pipeline.py"
    ).read_text(encoding="utf-8")

    assert "BEGIN FINAL SYNC PRODUCTION VERIFY V2" in app
    assert "BEGIN STREAM PRODUCTION ROUTE V2" in app
    assert "BEGIN FINAL PRODUCTION FAIL-CLOSED GATE V2" in pipeline
    assert "pipeline_sync(msg, hist)" in app
