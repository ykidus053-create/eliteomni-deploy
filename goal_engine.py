
import re, sqlite3, time, os
from threading import Lock

DB = os.path.expanduser("~/eliteomni_goals.db")
_lock = Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT, goal_text TEXT,
        status TEXT DEFAULT 'active',
        created_ts REAL, updated_ts REAL)''')
    con.commit()
    con.close()

_init()

_GOAL_PATTERNS = [
    r"(?:I want to|I need to|I am trying to|help me|I would like to)\s+(.+?)(?:\.|$)",
    r"(?:build|create|make|write|develop|implement)\s+(.+?)(?:\.|$)",
    r"(?:fix|debug|solve|resolve)\s+(.+?)(?:\.|$)",
]

def goal_detect_and_save(msg, session_id):
    for pat in _GOAL_PATTERNS:
        m = re.search(pat, msg, re.IGNORECASE)
        if m:
            goal_text = m.group(1).strip()[:200]
            if len(goal_text) > 15:
                try:
                    con = sqlite3.connect(DB)
                    existing = con.execute(
                        "SELECT id FROM goals WHERE session_id=? AND status='active' AND goal_text LIKE ?",
                        (session_id, goal_text[:40] + "%")).fetchone()
                    if not existing:
                        con.execute("INSERT INTO goals (session_id,goal_text,created_ts,updated_ts) VALUES (?,?,?,?)",
                                    (session_id, goal_text, time.time(), time.time()))
                        con.commit()
                    con.close()
                    return goal_text
                except Exception:
                    pass
    return ""

def goals_get_context(session_id="default"):
    try:
        con = sqlite3.connect(DB)
        rows = con.execute(
            "SELECT goal_text FROM goals WHERE session_id=? AND status='active' ORDER BY created_ts DESC LIMIT 3",
            (session_id,)).fetchall()
        con.close()
        if not rows:
            return ""
        return "[ACTIVE GOALS]: " + "; ".join(r[0] for r in rows)
    except Exception:
        return ""

def goal_complete(session_id, goal_text):
    try:
        con = sqlite3.connect(DB)
        con.execute("UPDATE goals SET status='completed', updated_ts=? WHERE session_id=? AND goal_text LIKE ?",
                    (time.time(), session_id, goal_text[:40] + "%"))
        con.commit()
        con.close()
    except Exception:
        pass
