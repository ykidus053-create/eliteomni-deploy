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

def wm_save(text: str, session_id: str = "default"):
    """Upgraded: Expanded regex patterns for better fact extraction."""
    patterns = {
        "user_name": r"my name is (\w+)",
        "user_role": r"I(?:'m| am) (?:a |an )?(\w+(?:\s\w+)?)",
        "user_work": r"I (?:work|am working) (?:at|for|on) ([\w\s]+?)(?:\.|,|$)",
        "user_pref": r"I (?:prefer|like|love|use|always use) ([\w\s]+?)(?:\.|,|$)",
        "user_location": r"I live in ([\w\s]+?)(?:\.|,|$)",
        "user_project": r"I am building ([\w\s]+?)(?:\.|,|$)"
    }
    with _lock:
        con = sqlite3.connect(DB)
        for key, pattern in patterns.items():
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                con.execute("INSERT OR REPLACE INTO facts (key,value,source,ts) VALUES (?,?,?,?)",
                            (key, val, "user_statement", time.time()))
        con.execute("INSERT OR REPLACE INTO memory (session_id,key,value,ts) VALUES (?,?,?,?)",
                    (session_id, "last_msg", text[:500], time.time()))
        con.commit()
        con.close()

def wm_retrieve(query: str, session_id: str = "default"):
    """Upgraded: Actually filters facts based on the query."""
    with _lock:
        con = sqlite3.connect(DB)
        facts = con.execute("SELECT key, value FROM facts ORDER BY ts DESC LIMIT 10").fetchall()
        recent = con.execute("SELECT value FROM memory WHERE session_id=? ORDER BY ts DESC LIMIT 5",
                             (session_id,)).fetchall()
        con.close()
        
    results = []
    if facts:
        q_lower = query.lower()
        # If query mentions a fact key or value, prioritize it
        relevant_facts = [f for f in facts if f[0].split('_')[1] in q_lower or f[1].lower() in q_lower]
        if not relevant_facts:
            relevant_facts = facts
        results.append("Known facts: " + "; ".join(f"{k}={v}" for k,v in relevant_facts))
    if recent:
        results.append("Recent context: " + " | ".join(r[0] for r in recent))
    return results

def wm_build_context(session_id: str = "default"):
    parts = wm_retrieve("", session_id)
    if not parts:
        return ""
    return "[PERSISTENT MEMORY]\n" + "\n".join(parts) + "\n[/PERSISTENT MEMORY]"

def wm_context(session_id: str = "default"):
    return wm_build_context(session_id)

def wm_update(session_id: str, user_msg: str, response: str):
    wm_save(user_msg, session_id)
