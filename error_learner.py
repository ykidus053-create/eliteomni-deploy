
import re, sqlite3, time, os
from threading import Lock

DB = os.path.expanduser("~/eliteomni_errors.db")
_lock = Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS error_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT, error_description TEXT,
        frequency INTEGER DEFAULT 1, skill TEXT, ts REAL)''')
    con.commit()
    con.close()

_init()

_ERROR_CHECKS = [
    ("off_by_one",       "range(len(",   "Off-by-one risk: prefer enumerate()"),
    ("bare_except",      "except:",      "Bare except catches KeyboardInterrupt too"),
    ("todo_placeholder", "TODO",         "Contains TODO placeholder -- incomplete"),
    ("fixme_marker",     "FIXME",        "Contains FIXME -- incomplete"),
]

def scan_for_errors(text, skill):
    errors = []
    for name, needle, desc in _ERROR_CHECKS:
        if needle in text:
            errors.append({"type": name, "description": desc})
    if "def " in text:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("def ") and ("=[]" in s or "={}" in s):
                errors.append({"type": "mutable_default", "description": "Mutable default argument"})
                break
    return errors

def record_error(error_type, skill, correction=""):
    try:
        con = sqlite3.connect(DB)
        existing = con.execute("SELECT id, frequency FROM error_patterns WHERE pattern_type=? AND skill=?",
                               (error_type, skill)).fetchone()
        if existing:
            con.execute("UPDATE error_patterns SET frequency=?, ts=? WHERE id=?",
                        (existing[1] + 1, time.time(), existing[0]))
        else:
            con.execute("INSERT INTO error_patterns (pattern_type,skill,ts) VALUES (?,?,?)",
                        (error_type, skill, time.time()))
        con.commit()
        con.close()
    except Exception:
        pass

def get_error_warnings(skill):
    try:
        con = sqlite3.connect(DB)
        rows = con.execute(
            "SELECT pattern_type, frequency FROM error_patterns WHERE skill=? ORDER BY frequency DESC LIMIT 3",
            (skill,)).fetchall()
        con.close()
        if not rows:
            return ""
        return "[KNOWN PITFALLS]\n" + "\n".join("Avoid: " + r[0] for r in rows)
    except Exception:
        return ""

def post_process_check(text, skill):
    errors = scan_for_errors(text, skill)
    for e in errors:
        record_error(e["type"], skill)
    if errors and skill == "coder":
        notes = "; ".join(e["description"] for e in errors[:2])
        text = text + "\n\n> Code review: " + notes
    return text

log_error = record_error

get_learned_corrections = get_error_warnings

detect_error_type = lambda text, skill: [e['type'] for e in scan_for_errors(text, skill)]
