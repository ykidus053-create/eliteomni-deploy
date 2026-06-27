import sqlite3, time, os, threading, logging
log = logging.getLogger(__name__)
DB_PATH = os.environ.get("MEMORY_DB", os.path.join(os.path.dirname(__file__), "data", "memory.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
_local = threading.local()

def _conn():
    if not hasattr(_local, "con"):
        _local.con = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
        _local.con.execute("PRAGMA journal_mode=WAL")
        _local.con.execute("""CREATE TABLE IF NOT EXISTS memory
                              (id INTEGER PRIMARY KEY, text TEXT, ts REAL, skill TEXT)""")
        _local.con.execute("""CREATE TABLE IF NOT EXISTS episodic
                              (id INTEGER PRIMARY KEY, summary TEXT, ts REAL)""")
        _local.con.execute("CREATE INDEX IF NOT EXISTS idx_memory_skill ON memory(skill)")
        _local.con.execute("CREATE INDEX IF NOT EXISTS idx_memory_text ON memory(text)")
        _local.con.commit()
    return _local.con

def mem_store(text: str, skill: str = "general"):
    try:
        with _conn() as con:
            con.execute("INSERT INTO memory (text, ts, skill) VALUES (?,?,?)", (text[:2000], time.time(), skill))
    except Exception as e:
        log.error("[mem_store] %s", e)

def mem_get(limit: int = 10, skill: str = None, query: str = None):
    """
    Upgraded: Now supports semantic keyword search.
    If a query is provided, it searches memory for matching text.
    """
    try:
        with _conn() as con:
            if query:
                # Basic keyword search fallback for vector-less memory
                search_pattern = f"%{query}%"
                if skill:
                    rows = con.execute(
                        "SELECT text FROM memory WHERE skill=? AND text LIKE ? ORDER BY ts DESC LIMIT ?",
                        (skill, search_pattern, limit)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "SELECT text FROM memory WHERE text LIKE ? ORDER BY ts DESC LIMIT ?",
                        (search_pattern, limit)
                    ).fetchall()
            elif skill:
                rows = con.execute("SELECT text FROM memory WHERE skill=? ORDER BY ts DESC LIMIT ?", (skill, limit)).fetchall()
            else:
                rows = con.execute("SELECT text FROM memory ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        log.error("[mem_get] %s", e); return []

def episodic_get(limit: int = 5):
    try:
        with _conn() as con:
            rows = con.execute("SELECT summary FROM episodic ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [r[0] for r in rows]
    except Exception as e:
        log.error("[episodic_get] %s", e); return []

def episodic_store(summary: str):
    try:
        with _conn() as con:
            con.execute("INSERT INTO episodic (summary, ts) VALUES (?,?)", (summary[:2000], time.time()))
    except Exception as e:
        log.error("[episodic_store] %s", e)

def stats():
    try:
        con = _conn()
        mem_count = con.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
        ep_count  = con.execute("SELECT COUNT(*) FROM episodic").fetchone()[0]
        return {"memory_rows": mem_count, "episodic_rows": ep_count, "db_path": DB_PATH}
    except Exception as e:
        return {"error": str(e)}
