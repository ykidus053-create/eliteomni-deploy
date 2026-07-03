import re, ast, sqlite3, time, os
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

def scan_for_errors(text: str, skill: str) -> list:
    """Upgraded: Uses AST parsing for perfect code analysis."""
    errors = []
    
    # Regex checks for non-code or simple patterns
    if "TODO" in text: errors.append({"type": "todo_placeholder", "description": "Contains TODO placeholder -- incomplete"})
    if "FIXME" in text: errors.append({"type": "fixme_marker", "description": "Contains FIXME -- incomplete"})
    if re.search(r'except\s*:', text): errors.append({"type": "bare_except", "description": "Bare except catches KeyboardInterrupt too"})

    # AST checks for robust code analysis
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for mutable default arguments
                for arg in node.args.defaults:
                    if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                        errors.append({"type": "mutable_default", "description": f"Mutable default argument in function {node.name}"})
                        break
                # Check for off-by-one in range(len())
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "range":
                        if child.args and isinstance(child.args[0], ast.Call) and isinstance(child.args[0].func, ast.Name) and child.args[0].func.id == "len":
                            errors.append({"type": "off_by_one", "description": "Off-by-one risk: prefer enumerate()"})
    except SyntaxError:
        pass # Not valid python code, skip AST checks

    return errors

def record_error(error_type, skill, correction=""):
    try:
        with _lock:
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
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute(
                "SELECT pattern_type, frequency FROM error_patterns WHERE skill=? ORDER BY frequency DESC LIMIT 3",
                (skill,)).fetchall()
            con.close()
        if not rows:
            return ""
        return "[KNOWN PITFALLS]\n" + "\n".join(f"Avoid: {r[0]} (seen {r[1]} times)" for r in rows)
    except Exception:
        return ""

def post_process_check(text, skill):
    errors = scan_for_errors(text, skill)
    for e in errors:
        record_error(e["type"], skill)
    if errors and skill == "coder":
        notes = "; ".join(e["description"] for e in errors[:2])
        text = text + "\n\n> ⚠️ Code review warning: " + notes
    return text

log_error = record_error
get_learned_corrections = get_error_warnings
detect_error_type = lambda text, skill: [e['type'] for e in scan_for_errors(text, skill)]
