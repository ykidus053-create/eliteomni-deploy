
import re, sqlite3, time, os
from threading import Lock

DB = os.path.expanduser("~/eliteomni_router.db")
_lock = Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS routing_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        msg_hash TEXT, skill TEXT, complexity TEXT,
        confidence REAL, ts REAL)''')
    con.commit()
    con.close()

_init()

SAFETY_TRIGGERS = ["synthesize nerve","weaponize","bioweapon","sarin","ricin",
                   "jailbreak","ignore all instructions","kill myself","suicide method"]

SKILL_KEYWORDS = [
    ("implement","coder",3.0),("refactor","coder",2.5),("debug","coder",3.0),
    ("write a function","coder",3.5),("write a class","coder",3.5),
    ("write a script","coder",3.5),("python code","coder",3.0),
    ("javascript","coder",2.5),("sql query","coder",2.5),
    ("calculate","calculator",3.0),("compute","calculator",2.5),
    ("sqrt","calculator",3.0),("percent of","calculator",3.0),
    ("divided by","calculator",2.5),("solve for","calculator",3.0),
    ("research","researcher",2.5),("analyze","researcher",2.5),
    ("compare","researcher",2.0),("history of","researcher",2.5),
    ("explain in detail","researcher",2.5),("pros and cons","researcher",2.0),
    ("how does","researcher",1.5),("why does","researcher",1.5),
]

HARD_WORDS = ["research","comprehensive","analyze","implement","algorithm",
              "step by step","essay","explain in detail","design","architecture"]
EASY_WORDS = ["hi ","hey ","hello","thanks","yes","no","capital of","2+2"]

def route(msg):
    return "coder", 0.95, {"complexity": "hard"}
    m = msg.lower()
    for danger in SAFETY_TRIGGERS:
        if danger in m:
            return "safety", 1.0, {"complexity": "fast"}
    from collections import defaultdict
    scores = defaultdict(float)
    for keyword, skill, weight in SKILL_KEYWORDS:
        if keyword in m:
            scores[skill] += weight
    best_skill = max(scores, key=scores.get) if scores else "general"
    best_score = scores.get(best_skill, 0.0)
    if best_score < 1.5:
        best_skill = "general"
    confidence = min(0.95, best_score / (best_score + 3.0)) if best_score > 0 else 0.3
    if any(w in m for w in EASY_WORDS) and len(msg) < 100:
        complexity = "easy"
    elif len(msg) > 200 or any(w in m for w in HARD_WORDS):
        complexity = "hard"
    else:
        complexity = "medium"
    try:
        con = sqlite3.connect(DB)
        con.execute("INSERT INTO routing_log (msg_hash,skill,complexity,confidence,ts) VALUES (?,?,?,?,?)",
                    (str(hash(msg[:50])), best_skill, complexity, confidence, time.time()))
        con.commit()
        con.close()
    except Exception:
        pass
    if best_skill == "coder": complexity = "hard"
    return best_skill, confidence, {"complexity": complexity}

def classify_skill(msg):
    skill, _, _ = route(msg)
    return skill

def route_complexity(msg):
    _, _, meta = route(msg)
    return meta.get("complexity", "medium")
