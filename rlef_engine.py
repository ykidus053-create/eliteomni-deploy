import sqlite3, os, re, time, threading, json
from collections import Counter

DB = os.path.expanduser("~/eliteomni_rlef.db")
_lock = threading.Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS execution_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_hash TEXT,
        error_type TEXT,
        error_trace TEXT,
        broken_code TEXT,
        fix_applied TEXT,
        success INTEGER,
        ts REAL
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON execution_traces(error_type)")
    con.commit(); con.close()
_init()

def _hash_task(task: str) -> str:
    import hashlib
    return hashlib.md5(task.encode()).hexdigest()[:12]

def record_execution_trace(task: str, error_trace: str, broken_code: str, fix_applied: str, success: bool):
    """Upgraded: Saves the exact (Error -> Fix) pair so the AI never repeats a mistake."""
    try:
        error_type = "Unknown"
        match = re.search(r'(\w+Error|Exception):', error_trace)
        if match: error_type = match.group(1)
        
        with _lock:
            con = sqlite3.connect(DB)
            con.execute("""INSERT INTO execution_traces 
                (task_hash, error_type, error_trace, broken_code, fix_applied, success, ts) 
                VALUES (?,?,?,?,?,?,?)""",
                (_hash_task(task), error_type, error_trace[:500], broken_code[:500], fix_applied[:500], int(success), time.time()))
            con.commit(); con.close()
    except: pass

def get_relevant_traces(error_trace: str, limit: int = 3) -> str:
    """Upgraded: When the AI hits an error, it retrieves past fixes for similar errors."""
    try:
        error_type = "Unknown"
        match = re.search(r'(\w+Error|Exception):', error_trace)
        if match: error_type = match.group(1)
        
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute("""SELECT error_trace, fix_applied, success FROM execution_traces 
                WHERE error_type=? AND success=1 ORDER BY ts DESC LIMIT ?""", (error_type, limit)).fetchall()
            con.close()
            
        if not rows: return ""
        
        context = f"[PAST EXECUTION TRACES FOR {error_type}]\n"
        context += "Here are EXACT fixes that worked for similar errors in the past:\n\n"
        for i, (err, fix, _) in enumerate(rows):
            context += f"Past Error {i+1}:\n{err[:200]}\nFix that worked:\n{fix[:300]}\n\n"
        context += "[END PAST TRACES]\n"
        return context
    except: return ""

def get_error_frequency() -> dict:
    """Returns the most common errors so the AI knows its weak points."""
    try:
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute("""SELECT error_type, COUNT(*) as n FROM execution_traces 
                GROUP BY error_type ORDER BY n DESC LIMIT 5""").fetchall()
            con.close()
        return {r[0]: r[1] for r in rows}
    except: return {}
