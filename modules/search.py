import urllib.request, urllib.parse, json, os

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")

def tool_search_multi(query, max_results=5):
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "engines": "duckduckgo,brave"})
        url = SEARXNG_URL + "/search?" + params
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read().decode())
        results = data.get("results", [])[:max_results]
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url_ = r.get("url", "")
            lines.append("[" + str(i) + "] " + title + "\n" + content + "\n" + url_)
        return "\n\n".join(lines)
    except Exception as e:
        print("[search] error:", e)
        return ""

def smart_search(query, history=None, max_results=5):
    return tool_search_multi(query, max_results=max_results)

# ── QUERY REWRITER (Huyen Ch.6 — rewrite ambiguous queries before retrieval) ──
def rewrite_query(query: str, history: list = None) -> str:
    """Rewrite ambiguous query into self-contained question using chat history."""
    try:
        from modules.core.http_client import mistral_generate
        history_str = ""
        if history:
            for m in history[-4:]:
                role = m.get("role", "")
                content = str(m.get("content", ""))[:200]
                history_str += f"{role}: {content}\n"
        msgs = [
            {"role": "system", "content": (
                "Rewrite the user query into a single self-contained search question. "
                "Use context from history if needed. Output ONLY the rewritten query, nothing else."
            )},
            {"role": "user", "content": f"History:\n{history_str}\nQuery: {query}"}
        ]
        rewritten = mistral_generate(msgs, max_tokens=100).strip()
        print(f"[query_rewrite] '{query}' → '{rewritten}'")
        return rewritten or query
    except Exception as e:
        print(f"[query_rewrite] failed: {e}")
        return query

# ── BM25 KEYWORD SEARCH ────────────────────────────────────────────────────────
def bm25_search(query: str, docs: list, top_k: int = 5) -> list:
    """
    Pure-Python BM25 over a list of doc strings (Huyen Ch.6 — hybrid search).
    Returns top_k docs sorted by BM25 score.
    """
    import math
    k1, b = 1.5, 0.75
    tokenize = lambda t: t.lower().split()
    tokenized = [tokenize(d) for d in docs]
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    query_terms = tokenize(query)
    scores = []
    N = len(docs)
    for i, tokens in enumerate(tokenized):
        tf_map = {}
        for tok in tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            df = sum(1 for t in tokenized if term in t)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(tokens) / avgdl))
            score += idf * tf_norm
        scores.append((score, docs[i]))
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

# ── HYBRID SEARCH (BM25 + semantic via SearXNG) ───────────────────────────────
def hybrid_search(query: str, history: list = None, docs: list = None, max_results: int = 5) -> str:
    """
    Huyen Ch.6: BM25 for speed + web semantic search for accuracy.
    Rewrites query first, then merges both result sets.
    """
    rewritten = rewrite_query(query, history)

    # Web results (semantic)
    web = tool_search_multi(rewritten, max_results=max_results)
    web_results = web.split("\n\n") if web else []

    # BM25 over local docs if provided
    bm25_results = []
    if docs:
        bm25_results = bm25_search(rewritten, docs, top_k=max_results)

    # Merge: interleave BM25 and web, deduplicate
    seen = set()
    merged = []
    for r in bm25_results + web_results:
        key = r[:80]
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return "\n\n".join(merged[:max_results]) or ""
