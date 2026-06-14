"""
In-memory token budget counter — flushes to SQLite every 60s.
Replaces per-request SQLite write in _budget().
"""
import threading, time, sqlite3, os

_DB_PATH = os.path.expanduser("~/eliteomni_memory.db")

class BudgetCounter:
    def __init__(self, flush_interval: int = 60):
        self._lock   = threading.Lock()
        self._counts = {}          # {user_id: tokens_used}
        self._dirty  = False
        self._interval = flush_interval
        threading.Thread(target=self._flush_loop, daemon=True,
                         name="budget_flush").start()

    def record(self, user_id: str, tokens: int):
        with self._lock:
            self._counts[user_id] = self._counts.get(user_id, 0) + tokens
            self._dirty = True

    def get(self, user_id: str) -> int:
        with self._lock:
            return self._counts.get(user_id, 0)

    def _flush_loop(self):
        while True:
            time.sleep(self._interval)
            self._flush()

    def _flush(self):
        with self._lock:
            if not self._dirty: return
            snapshot = dict(self._counts)
            self._dirty = False
        try:
            con = sqlite3.connect(_DB_PATH)
            con.execute("""CREATE TABLE IF NOT EXISTS budget_log
                           (user_id TEXT PRIMARY KEY, tokens INTEGER, ts REAL)""")
            for uid, tok in snapshot.items():
                con.execute(
                    "INSERT OR REPLACE INTO budget_log (user_id,tokens,ts) VALUES (?,?,?)",
                    (uid, tok, time.time())
                )
            con.commit(); con.close()
        except Exception as e:
            print(f"[BudgetCounter] flush error: {e}")

budget_counter = BudgetCounter()
