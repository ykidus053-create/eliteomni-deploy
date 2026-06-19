
import re, sqlite3, time, os
from threading import Lock

DB = os.path.expanduser("~/eliteomni_memory.db")
_lock = Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        key TEXT,
        value TEXT,
        source TEXT DEFAULT 'inferred',
        ts REAL,
        UNIQUE(session_id, key))''')
    con.execute('''CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE,
        value TEXT,
        confidence REAL DEFAULT 0.8,
        source TEXT,
        ts REAL)''')
    con.commit()
    con.close()

_init()

def wm_save(text, session_id="default"):
    name_m = re.search(r"my name is (\w+)", text, re.IGNORECASE)
    role_m = re.search(r"I(?:'m| am) (?:a |an )?(\w+(?:\s\w+)?)", text, re.IGNORECASE)
    work_m = re.search(r"I (?:work|am working) (?:at|for|on) ([\w\s]+?)(?:\.|,|$)", text, re.IGNORECASE)
    pref_m = re.search(r"I (?:prefer|like|love|use|always use) ([\w\s]+?)(?:\.|,|$)", text, re.IGNORECASE)
    with _lock:
        con = sqlite3.connect(DB)
        for key, m in [("user_name", name_m), ("user_role", role_m), ("user_work", work_m), ("user_pref", pref_m)]:
            if m:
                val = m.group(1).strip()
                con.execute("INSERT OR REPLACE INTO facts (key,value,source,ts) VALUES (?,?,?,?)",
                            (key, val, "user_statement", time.time()))
        con.execute("INSERT OR REPLACE INTO memory (session_id,key,value,ts) VALUES (?,?,?,?)",
                    (session_id, "last_msg", text[:300], time.time()))
        con.commit()
        con.close()

def wm_retrieve(query, session_id="default"):
    with _lock:
        con = sqlite3.connect(DB)
        facts = con.execute("SELECT key, value FROM facts ORDER BY ts DESC LIMIT 10").fetchall()
        recent = con.execute("SELECT value FROM memory WHERE session_id=? ORDER BY ts DESC LIMIT 5",
                             (session_id,)).fetchall()
        con.close()
    results = []
    if facts:
        results.append("Known facts: " + "; ".join(f"{k}={v}" for k,v in facts))
    if recent:
        results.append("Recent context: " + " | ".join(r[0] for r in recent))
    return results

def wm_build_context(session_id="default"):
    parts = wm_retrieve("", session_id)
    if not parts:
        return ""
    return "[PERSISTENT MEMORY]\n" + "\n".join(parts) + "\n[/PERSISTENT MEMORY]"

def wm_context(session_id="default"):
    return wm_build_context(session_id)

def wm_update(session_id, user_msg, response):
    wm_save(user_msg, session_id)
