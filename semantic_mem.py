"""
Pure-Python Semantic Memory (TF-IDF + Cosine Similarity).
Upgraded: Created the missing file! Provides vector-like semantic search
without requiring heavy dependencies like FAISS or PyTorch.
"""
import sqlite3, time, os, re, math
from threading import Lock
from collections import Counter

DB = os.path.expanduser("~/eliteomni_semantic.db")
_lock = Lock()

def _init():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        ts REAL
    )''')
    con.commit()
    con.close()
_init()

def _tokenize(text):
    return re.findall(r'\b\w{3,}\b', text.lower())

def _tf(tokens):
    count = Counter(tokens)
    total = len(tokens)
    return {word: count[word] / total for word in count} if total > 0 else {}

def _idf(corpus_tokens):
    N = len(corpus_tokens)
    df = Counter()
    for tokens in corpus_tokens:
        for word in set(tokens):
            df[word] += 1
    return {word: math.log(N / (df[word] + 1)) for word in df}

def _vectorize(text, idf_scores):
    tokens = _tokenize(text)
    tf = _tf(tokens)
    return {word: tf[word] * idf_scores.get(word, 0) for word in tf}

def _cosine_sim(vec1, vec2):
    if not vec1 or not vec2: return 0.0
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(v**2 for v in vec1.values())
    sum2 = sum(v**2 for v in vec2.values())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0.0

def semantic_store(text: str):
    if not text or len(text) < 15: return
    try:
        with _lock:
            con = sqlite3.connect(DB)
            con.execute("INSERT INTO memories (text, ts) VALUES (?,?)", (text[:2000], time.time()))
            con.commit()
            con.close()
    except Exception:
        pass

def semantic_retrieve(query: str, limit: int = 3) -> list:
    """Retrieves semantically similar memories using TF-IDF Cosine Similarity."""
    try:
        with _lock:
            con = sqlite3.connect(DB)
            rows = con.execute("SELECT text FROM memories ORDER BY ts DESC LIMIT 500").fetchall()
            con.close()
            
        if not rows: return []
        
        texts = [r[0] for r in rows]
        corpus_tokens = [_tokenize(t) for t in texts]
        idf_scores = _idf(corpus_tokens)
        
        query_vec = _vectorize(query, idf_scores)
        scored = []
        for i, text in enumerate(texts):
            text_vec = _vectorize(text, idf_scores)
            sim = _cosine_sim(query_vec, text_vec)
            if sim > 0.1:
                scored.append((sim, text))
                
        scored.sort(key=lambda x: -x[0])
        return [text for _, text in scored[:limit]]
    except Exception:
        return []

def semantic_context(query: str) -> str:
    results = semantic_retrieve(query, limit=3)
    if not results: return ""
    return "[SEMANTIC MEMORIES]\n" + "\n- ".join(results) + "\n[/SEMANTIC MEMORIES]"
