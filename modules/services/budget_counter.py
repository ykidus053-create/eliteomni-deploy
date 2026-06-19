"""
In-memory budget counter with periodic SQLite flush.
Ng: removes per-request SQLite write from hot path.
Previously: sqlite3.connect + INSERT + DELETE on every single inference call.
Now: in-memory increment, flush every 60 seconds.
"""
import time, threading, sqlite3, os
from collections import defaultdict

_DB_PATH   = os.path.expanduser("~/eliteomni_memory.db")
_counts: dict = defaultdict(lambda: {"calls": 0, "tokens": 0, "budget": 0})
_lock      = threading.Lock()
_last_flush = time.time()
_FLUSH_INTERVAL = 60

def record_budget(skill: str, complexity: str, msg_len: int, budget: int) -> None:
    """Record budget usage in memory. Flush to SQLite every 60s."""
    global _last_flush
    key = f"{skill}:{complexity}"
    with _lock:
        _counts[key]["calls"]  += 1
        _counts[key]["budget"] += budget
        _counts[key]["tokens"] += msg_len // 4
    now = time.time()
    if now - _last_flush > _FLUSH_INTERVAL:
        _flush_to_db()
        _last_flush = now

def _flush_to_db() -> None:
    try:
        con = sqlite3.connect(_DB_PATH)
        con.execute("""CREATE TABLE IF NOT EXISTS budget_summary
            (key TEXT PRIMARY KEY, calls INTEGER, budget INTEGER, tokens INTEGER, ts REAL)""")
        with _lock:
            snapshot = dict(_counts)
        for key, vals in snapshot.items():
            con.execute("""INSERT INTO budget_summary (key,calls,budget,tokens,ts)
                VALUES (?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET
                calls=calls+excluded.calls,
                budget=budget+excluded.budget,
                tokens=tokens+excluded.tokens,
                ts=excluded.ts""",
                (key, vals["calls"], vals["budget"], vals["tokens"], time.time()))
        con.commit(); con.close()
    except Exception as e:
        print(f"[BudgetFlush] {e}")
