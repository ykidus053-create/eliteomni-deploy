"""
Knowledge RAG — embeds all book-gap functions and injects relevant ones
into the system prompt at query time. This is what makes the book knowledge
actually affect model responses.
"""
import os, sys, ast, inspect, importlib, json, math, sqlite3, time, threading

_DB = os.path.expanduser("~/eliteomni_knowledge.db")
_lock = threading.Lock()
_cache: dict = {}  # query -> chunks

BOOK_MODULES = [
    "book_gaps_impl", "book8_gaps", "final_gaps", "aie_book_impl",
    "dl_book_implementations", "dl_book_implementations2",
    "dl_book_implementations3", "goodfellow_dl", "gaps_all_books",
    "math_impl",
]

def _init_db():
    con = sqlite3.connect(_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS knowledge (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        module TEXT, name TEXT, kind TEXT,
        doc TEXT, signature TEXT, chunk TEXT
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_module ON knowledge(module)")
    con.commit(); con.close()
_init_db()

def _extract_chunks(module_name: str) -> list:
    """Extract function/class name + docstring + signature as knowledge chunks."""
    chunks = []
    try:
        if module_name in sys.modules:
            mod = sys.modules[module_name]
        else:
            mod = importlib.import_module(module_name)
        for name in dir(mod):
            if name.startswith("_"): continue
            obj = getattr(mod, name, None)
            if obj is None: continue
            kind = None
            if callable(obj) and hasattr(obj, "__doc__"):
                kind = "class" if isinstance(obj, type) else "function"
            if not kind: continue
            doc = (obj.__doc__ or "").strip()[:300]
            try:
                sig = str(inspect.signature(obj))[:150]
            except:
                sig = ""
            chunk = f"{kind} {name}{sig}: {doc}"
            chunks.append({"module": module_name, "name": name,
                           "kind": kind, "doc": doc,
                           "signature": sig, "chunk": chunk})
    except Exception as e:
        print(f"[knowledge_rag] extract error {module_name}: {e}")
    return chunks

def build_index(force: bool = False):
    """Build/rebuild the knowledge index from all book modules."""
    con = sqlite3.connect(_DB)
    count = con.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
    con.close()
    if count > 0 and not force:
        print(f"[knowledge_rag] index exists ({count} chunks) — skipping rebuild")
        return count

    con = sqlite3.connect(_DB)
    con.execute("DELETE FROM knowledge")
    total = 0
    for mod_name in BOOK_MODULES:
        chunks = _extract_chunks(mod_name)
        for c in chunks:
            con.execute(
                "INSERT INTO knowledge(module,name,kind,doc,signature,chunk) VALUES(?,?,?,?,?,?)",
                (c["module"], c["name"], c["kind"], c["doc"], c["signature"], c["chunk"])
            )
        total += len(chunks)
        if chunks:
            print(f"[knowledge_rag] indexed {mod_name}: {len(chunks)} chunks")
    con.commit(); con.close()
    print(f"[knowledge_rag] ✅ index built: {total} total chunks")
    return total

def _bm25_score(query: str, chunk: str) -> float:
    """BM25-inspired scoring."""
    k1, b = 1.5, 0.75
    q_words = query.lower().split()
    c_words = chunk.lower().split()
    if not q_words or not c_words: return 0.0
    avg_len = 80
    score = 0.0
    for w in q_words:
        tf = c_words.count(w)
        if tf == 0: continue
        idf = math.log(1 + 1 / (0.5 + tf))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(c_words) / avg_len))
    return score

def _semantic_score(query: str, chunk: str) -> float:
    """Cosine similarity via embeddings."""
    try:
        from modules.services.semantic_mem import _embedder
        if not _embedder: return 0.0
        import numpy as np
        q_emb = list(_embedder.embed([query]))[0]
        c_emb = list(_embedder.embed([chunk]))[0]
        q, c = np.array(q_emb), np.array(c_emb)
        return float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-8))
    except Exception:
        return 0.0

def _expand_query(query: str) -> list:
    """Query expansion for better recall."""
    q = query.lower().strip()
    variants = [q]
    expansions = {"ml": "machine learning", "dl": "deep learning",
                  "nlp": "natural language processing", "rl": "reinforcement learning",
                  "llm": "large language model", "rag": "retrieval augmented generation",
                  "nn": "neural network", "cnn": "convolutional neural network"}
    for k, v in expansions.items():
        if k in q.split(): variants.append(q.replace(k, v))
        if v in q: variants.append(q.replace(v, k))
    return list(set(variants))

def retrieve(query: str, top_k: int = 8, min_score: float = 0.05) -> list:
    """Hybrid BM25 + semantic retrieval with MMR reranking."""
    cache_key = f"{query}:{top_k}"
    if cache_key in _cache:
        return _cache[cache_key]
    try:
        con = sqlite3.connect(_DB)
        rows = con.execute("SELECT name, kind, doc, signature, chunk, module FROM knowledge").fetchall()
        con.close()
    except:
        return []
    if not rows:
        return []
    queries = _expand_query(query)
    scored = {}
    for row in rows:
        name, kind, doc, sig, chunk, module = row
        key = name + module
        bm25 = max(_bm25_score(q, chunk) for q in queries)
        sem = _semantic_score(query, chunk)
        hybrid = 0.6 * sem + 0.4 * (bm25 / (bm25 + 1))
        if any(w in name.lower() for w in query.lower().split()):
            hybrid *= 1.5
        if kind == "class": hybrid *= 1.1
        if len(doc or "") > 100: hybrid *= 1.1
        if hybrid >= min_score:
            scored[key] = (hybrid, name, kind, doc, sig, module)
    results = sorted(scored.values(), reverse=True)[:top_k * 2]
    # MMR reranking for diversity
    if results:
        selected = [results[0]]
        remaining = results[1:]
        while remaining and len(selected) < top_k:
            best, best_idx = None, 0
            for i, r in enumerate(remaining):
                diversity = 1.0 if r[5] not in [s[5] for s in selected] else 0.7
                score = r[0] * diversity
                if best is None or score > best:
                    best, best_idx = score, i
            selected.append(remaining.pop(best_idx))
        results = selected
    _cache[cache_key] = results
    if len(_cache) > 500:
        _cache.pop(next(iter(_cache)))
    return results

def get_knowledge_context(query: str, top_k: int = 8) -> str:
    """Get formatted knowledge context to inject into system prompt."""
    results = retrieve(query, top_k=top_k)
    if not results:
        return ""
    lines = ["[RELEVANT KNOWLEDGE FROM ML/DL BOOKS]"]
    for score, name, kind, doc, sig, module in results:
        lines.append(f"• {kind} `{name}{sig}` ({module}): {doc[:150]}")
    lines.append("[END KNOWLEDGE]")
    return "\n".join(lines)

def start_background_indexer():
    """Rebuild index in background on startup, then re-index every 5 min."""
    def _run():
        time.sleep(2)  # let app finish starting
        build_index(force=True)
        while True:
            time.sleep(300)  # re-index every 5 min to catch hot-reloaded changes
            build_index(force=True)
    t = threading.Thread(target=_run, daemon=True, name="knowledge_indexer")
    t.start()

if __name__ == "__main__":
    build_index(force=True)
    print(retrieve("gradient descent optimization", top_k=3))
