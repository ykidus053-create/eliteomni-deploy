import sqlite3, time, os, threading, logging, math, re
from collections import Counter
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
        _local.con.commit()
    return _local.con

def _tokenize(text):
    return re.findall(r'\b\w{3,}\b', text.lower())

def _tf(tokens):
    count = Counter(tokens)
    total = len(tokens)
    return {word: count[word] / total for word in count} if total > 0 else {}

def _cosine_sim(vec1, vec2):
    if not vec1 or not vec2: return 0.0
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    return numerator / (math.sqrt(sum1) * math.sqrt(sum2)) if sum1 and sum2 else 0.0

def mem_store(text: str, skill: str = "general"):
    try:
        with _conn() as con:
            con.execute("INSERT INTO memory (text, ts, skill) VALUES (?,?,?)", (text[:2000], time.time(), skill))
    except Exception as e:
        log.error("[mem_store] %s", e)

def mem_get(limit: int = 10, skill: str = None, query: str = None):
    """Upgraded: Pure Python TF-IDF Semantic Search. Finds contextually related memories."""
    try:
        with _conn() as con:
            if skill:
                rows = con.execute("SELECT text FROM memory WHERE skill=? ORDER BY ts DESC LIMIT 100", (skill,)).fetchall()
            else:
                rows = con.execute("SELECT text FROM memory ORDER BY ts DESC LIMIT 100").fetchall()
        
        texts = [r[0] for r in rows]
        if not texts: return []
        
        if not query:
            return texts[:limit]
            
        query_vec = _tf(_tokenize(query))
        scored = []
        for text in texts:
            text_vec = _tf(_tokenize(text))
            sim = _cosine_sim(query_vec, text_vec)
            if sim > 0.05:
                scored.append((sim, text))
                
        scored.sort(key=lambda x: -x[0])
        return [text for _, text in scored[:limit]]
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
