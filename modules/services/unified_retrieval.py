# UNIFIED RETRIEVAL — Fei-Fei Li suggestion
# Semantic-first, keyword fallback, never silently degrades

def unified_retrieve(query: str, k: int = 5) -> list:
    """
    Try semantic retrieval first (chromadb), fall back to keyword (SQLite).
    Returns unified list ranked by relevance. Never raises.
    """
    results = []
    # Tier 1: semantic (chromadb + fastembed)
    try:
        from modules.services.semantic_mem import semantic_mem_get
        sem = semantic_mem_get(query, k=k)
        results.extend([{"text": t, "source": "semantic", "score": 1.0} for t in sem])
    except Exception as e:
        print(f"[UnifiedRetrieval] semantic failed: {e}")
    # Tier 2: keyword SQLite fallback
    if len(results) < k:
        try:
            from modules.services.memory import db_mem_get
            kw = db_mem_get(query, k=k - len(results))
            results.extend([{"text": t, "source": "keyword", "score": 0.6} for t in kw])
        except Exception as e:
            print(f"[UnifiedRetrieval] keyword failed: {e}")
    # Deduplicate
    seen, deduped = set(), []
    for r in results:
        key = r["text"][:80]
        if key not in seen:
            seen.add(key); deduped.append(r)
    return deduped[:k]
