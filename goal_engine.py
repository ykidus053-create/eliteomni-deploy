import re, sqlite3, time, os, json
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
    con.execute('''CREATE TABLE IF NOT EXISTS sub_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        goal_id INTEGER, task_text TEXT,
        status TEXT DEFAULT 'pending',
        created_ts REAL)''')
    con.commit()
    con.close()

_init()

_GOAL_PATTERNS = [
    r"(?:I want to|I need to|I am trying to|help me|I would like to)\s+(.+?)(?:\.|$)",
    r"(?:build|create|make|write|develop|implement)\s+(.+?)(?:\.|$)",
    r"(?:fix|debug|solve|resolve)\s+(.+?)(?:\.|$)",
]

def goal_detect_and_save(msg, session_id, generate_fn=None, model="mistral-medium-latest"):
    """Upgraded: Detects goal and uses LLM to decompose it into sub-tasks."""
    for pat in _GOAL_PATTERNS:
        m = re.search(pat, msg, re.IGNORECASE)
        if m:
            goal_text = m.group(1).strip()[:200]
            if len(goal_text) > 15:
                try:
                    with _lock:
                        con = sqlite3.connect(DB)
                        existing = con.execute(
                            "SELECT id FROM goals WHERE session_id=? AND status='active' AND goal_text LIKE ?",
                            (session_id, goal_text[:40] + "%")).fetchone()
                        if not existing:
                            con.execute("INSERT INTO goals (session_id,goal_text,created_ts,updated_ts) VALUES (?,?,?,?)",
                                        (session_id, goal_text, time.time(), time.time()))
                            con.commit()
                            goal_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
                            
                            # Upgraded: LLM Decomposition
                            if generate_fn and goal_id:
                                try:
                                    prompt = [f"Break this goal into 3-5 short, actionable sub-tasks. Output ONLY a JSON list of strings: {goal_text}"]
                                    raw = generate_fn(prompt, max_tokens=300, model=model)
                                    raw = re.sub(r'```json|```', '', raw).strip()
                                    tasks = json.loads(raw)
                                    for t in tasks:
                                        con.execute("INSERT INTO sub_tasks (goal_id, task_text, created_ts) VALUES (?,?,?)",
                                                    (goal_id, str(t)[:150], time.time()))
                                    con.commit()
                                except:
                                    pass
                        con.close()
                    return goal_text
                except Exception:
                    pass
    return ""

def goals_get_context(session_id="default"):
    try:
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute(
                "SELECT id, goal_text FROM goals WHERE session_id=? AND status='active' ORDER BY created_ts DESC LIMIT 1",
                (session_id,)).fetchall()
            if not rows: return ""
            goal_id, goal_text = rows[0]
            tasks = con.execute("SELECT task_text, status FROM sub_tasks WHERE goal_id=? ORDER BY created_ts ASC", (goal_id,)).fetchall()
            con.close()
            
        ctx = f"[ACTIVE GOAL]: {goal_text}\n"
        if tasks:
            for i, (t, status) in enumerate(tasks, 1):
                icon = "✅" if status == "completed" else "⬜"
                ctx += f"{icon} {i}. {t}\n"
        return ctx
    except Exception:
        return ""

def goal_complete(session_id, goal_text):
    try:
        with _lock:
            con = sqlite3.connect(DB)
            con.execute("UPDATE goals SET status='completed', updated_ts=? WHERE session_id=? AND goal_text LIKE ?",
                        (time.time(), session_id, goal_text[:40] + "%"))
            con.commit()
            con.close()
    except Exception:
        pass
