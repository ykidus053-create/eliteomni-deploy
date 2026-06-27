import re, sqlite3, time, os, threading

DB = os.path.expanduser("~/eliteomni_reflection.db")
_lock = threading.Lock()

def _init_db():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lesson TEXT UNIQUE,
        skill TEXT,
        ts REAL
    )''')
    con.commit(); con.close()
_init_db()

def should_reflect(msg, skill, complexity):
    if complexity == "easy": return False
    triggers = {
        "coder": ["implement","debug","build","write","fix"],
        "researcher": ["analyze","compare","explain","why","how does"],
        "calculator": ["calculate","compute","solve","percent","formula"],
    }
    t = triggers.get(skill, [])
    return complexity in ("medium","hard") and (any(x in msg.lower() for x in t) or complexity == "hard")

def reflect_on_response(response, msg, skill):
    issues = []
    if skill == "coder":
        if "TODO" in response or "FIXME" in response: issues.append("Contains TODO/FIXME placeholder -- incomplete")
        code_words = ["implement","write","code","function","script","class"]
        if any(kw in msg.lower() for kw in code_words):
            if "```" not in response and "def " not in response and "class " not in response: issues.append("Coding task but no code block found")
    if skill == "calculator":
        if not re.search(r"[\d\.]+", response): issues.append("Calculator task but no numeric answer found")
    return len(issues) == 0, issues

def annotate_response(response, issues, skill):
    if not issues or skill in ("general","safety"): return response
    return response + "\n\n> Self-check: " + " | ".join(issues[:2])

def build_reflection_prompt(msg, response, issues, skill):
    if not issues: return ""
    return ("Your previous response had these issues:\n" + "\n".join("- " + i for i in issues) +
            "\n\nOriginal question: " + msg[:200] + "\nPlease provide a complete, corrected response.")

# Upgraded: Episodic Memory Buffer
def consolidate_lessons(history: list, generate_fn, model: str, skill: str = "general"):
    """Extract generalized lessons from a completed conversation and store them."""
    if not history or len(history) < 4: return
    try:
        convo_str = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in history[-6:]])
        prompt = [
            {"role": "system", "content": "Analyze this conversation and extract 1-2 generalized, actionable lessons for future interactions. Reply ONLY in JSON format: {\"lessons\": [\"...\", \"...\"]}"},
            {"role": "user", "content": convo_str}
        ]
        raw = generate_fn(prompt, max_tokens=200, model=model)
        import json, re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            with _lock:
                con = sqlite3.connect(DB)
                for lesson in data.get("lessons", []):
                    try:
                        con.execute("INSERT INTO lessons (lesson, skill, ts) VALUES (?,?,?)", (lesson[:200], skill, time.time()))
                    except sqlite3.IntegrityError:
                        pass # Duplicate lesson
                con.commit(); con.close()
    except Exception:
        pass

def get_episodic_lessons(skill: str = "general", limit: int = 3) -> str:
    """Retrieve past lessons to inject into the system prompt."""
    try:
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute("SELECT lesson FROM lessons WHERE skill=? OR skill='general' ORDER BY ts DESC LIMIT ?", (skill, limit)).fetchall()
            con.close()
        if not rows: return ""
        return "[PAST LESSONS LEARNED]\n" + "\n".join(f"- {r[0]}" for r in rows) + "\n[/PAST LESSONS]"
    except Exception:
        return ""
