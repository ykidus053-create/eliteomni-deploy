"""
Thread-local SQLite connection pool.
Drop-in replacement for all inline sqlite3.connect() calls.
Swap backend to Postgres later with zero app changes.
"""
import sqlite3, threading, os, time

_DB_PATH  = os.path.expanduser("~/eliteomni_memory.db")
_FT_PATH  = os.path.expanduser("~/eliteomni_finetune.db")
_local    = threading.local()

def _conn(path: str) -> sqlite3.Connection:
    """Return a thread-local connection; create once per thread."""
    attr = f"_conn_{hash(path) & 0xFFFFFF}"
    conn = getattr(_local, attr, None)
    if conn is None:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")   # 8 MB page cache
        setattr(_local, attr, conn)
    return conn

def mem_conn()    -> sqlite3.Connection: return _conn(_DB_PATH)
def finetune_conn() -> sqlite3.Connection: return _conn(_FT_PATH)

class Repository:
    """Thin data-access layer over SQLite. Swap to Postgres: change _conn only."""

    # ── memory ──────────────────────────────────────────────────────────────
    def mem_save(self, text: str, source: str = "conversation"):
        c = mem_conn()
        c.execute("INSERT INTO memory (text,source,ts) VALUES (?,?,?)",
                  (text[:1000], source, time.time()))
        c.execute("DELETE FROM memory WHERE id NOT IN "
                  "(SELECT id FROM memory ORDER BY ts DESC LIMIT 5000)")
        c.commit()

    def mem_get(self, query: str, k: int = 5) -> list:
        import re
        kws = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        if not kws: return []
        rows = mem_conn().execute(
            "SELECT text FROM memory ORDER BY ts DESC LIMIT 500"
        ).fetchall()
        scored = sorted(
            [(sum(1 for w in kws if w in t.lower()), t) for (t,) in rows if any(w in t.lower() for w in kws)],
            reverse=True
        )
        return [t for _, t in scored[:k]]

    # ── episodic ─────────────────────────────────────────────────────────────
    def episodic_save(self, text: str):
        c = mem_conn()
        c.execute("INSERT INTO episodic (text,ts) VALUES (?,?)",
                  (text[:500], time.time()))
        c.execute("DELETE FROM episodic WHERE id NOT IN "
                  "(SELECT id FROM episodic ORDER BY ts DESC LIMIT 200)")
        c.commit()

    def episodic_get(self, query: str, k: int = 5) -> list:
        import re
        kws = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
        rows = mem_conn().execute(
            "SELECT text FROM episodic ORDER BY ts DESC LIMIT 200"
        ).fetchall()
        scored = sorted(
            [(sum(1 for w in kws if w in t.lower()), t) for (t,) in rows if any(w in t.lower() for w in kws)],
            reverse=True
        )
        return [t for _, t in scored[:k]]

    # ── kv ───────────────────────────────────────────────────────────────────
    def kv_set(self, key: str, value: str):
        c = mem_conn()
        c.execute("INSERT OR REPLACE INTO kv (key,value,ts) VALUES (?,?,?)",
                  (key, value, time.time()))
        c.commit()

    def kv_get(self, key: str) -> str:
        row = mem_conn().execute(
            "SELECT value FROM kv WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else ""

    # ── memory stats ─────────────────────────────────────────────────────────
    def mem_stats(self) -> dict:
        c = mem_conn()
        return {
            "memory":  c.execute("SELECT COUNT(*) FROM memory").fetchone()[0],
            "episodic": c.execute("SELECT COUNT(*) FROM episodic").fetchone()[0],
            "kv":      c.execute("SELECT COUNT(*) FROM kv").fetchone()[0],
            "recent":  [r[0] for r in c.execute(
                        "SELECT text FROM memory ORDER BY ts DESC LIMIT 3").fetchall()],
        }

    def mem_clear(self):
        c = mem_conn()
        c.execute("DELETE FROM memory")
        c.execute("DELETE FROM episodic")
        c.commit()

    # ── finetune ─────────────────────────────────────────────────────────────
    def finetune_save(self, skill, complexity, system, user, response, rating=0):
        from datetime import datetime, timezone
        c = finetune_conn()
        c.execute(
            "INSERT INTO samples (ts,skill,complexity,system_prompt,user_msg,assistant_response,rating) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(),
             skill, complexity, system[:800], user[:600], response[:1200], rating)
        )
        c.commit()

    def finetune_stats(self) -> dict:
        c = finetune_conn()
        return {
            "total": c.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
            "rated": c.execute("SELECT COUNT(*) FROM samples WHERE rating>0").fetchone()[0],
            "by_skill": dict(c.execute(
                "SELECT skill, COUNT(*) FROM samples GROUP BY skill").fetchall()),
        }

repo = Repository()   # module singleton
