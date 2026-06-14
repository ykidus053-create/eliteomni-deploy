"""
JIT (Just-In-Time) Retrieval
Detects knowledge gaps mid-reasoning and fetches relevant context on demand.
Gaps detected by: uncertainty markers, question patterns, entity mentions.
"""
import re, time, threading, sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

DB = Path.home() / "eliteomni_jit.db"
_lock = threading.Lock()

UNCERTAINTY_SIGNALS = [
    r"\b(I think|I believe|I'm not sure|not certain|approximately|around|roughly)\b",
    r"\b(may have|might be|could be|probably|likely|unclear)\b",
    r"\b(as of my knowledge|my training|I recall|I remember)\b",
    r"\?\s*$",
    r"\b(verify|check|confirm|source)\b",
]

KNOWLEDGE_GAP_SIGNALS = [
    r"\b(I don't know|I cannot find|I lack|outside my knowledge)\b",
    r"\b(recent|latest|current|updated|new|2025|2026)\b",
    r"\b(specific|exact|precise)\s+(number|date|version|price|rate)\b",
]

def _conn():
    c = sqlite3.connect(str(DB))
    c.execute("""CREATE TABLE IF NOT EXISTS jit_cache (
        query_hash TEXT PRIMARY KEY,
        query TEXT, result TEXT, source TEXT,
        confidence REAL, ts REAL, hits INTEGER DEFAULT 1)""")
    c.execute("""CREATE TABLE IF NOT EXISTS gap_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signal TEXT, context_snippet TEXT,
        retrieval_triggered INTEGER, ts REAL)""")
    c.commit()
    return c

def detect_knowledge_gaps(partial_response: str, original_msg: str) -> List[Tuple[str, str]]:
    """
    Returns list of (gap_type, query_to_fetch) pairs.
    Called mid-generation or pre-generation on complex queries.
    """
    gaps = []
    text = partial_response + " " + original_msg

    for pattern in UNCERTAINTY_SIGNALS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            context = text[max(0, m.start()-50):m.end()+100].strip()
            entity = _extract_entity_near(text, m.start())
            if entity:
                gaps.append(("uncertainty", entity))

    for pattern in KNOWLEDGE_GAP_SIGNALS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            context = text[max(0, m.start()-30):m.end()+80].strip()
            gaps.append(("knowledge_gap", context[:120]))

    seen = set()
    deduped = []
    for g in gaps:
        k = g[1][:40].lower()
        if k not in seen:
            seen.add(k)
            deduped.append(g)
    return deduped[:3]

def _extract_entity_near(text: str, pos: int) -> Optional[str]:
    window = text[max(0, pos-80):pos+120]
    candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', window)
    if candidates:
        return candidates[0]
    words = window.split()
    if len(words) >= 3:
        return " ".join(words[1:4])
    return None

def jit_fetch(query: str, search_fn) -> Optional[str]:
    import hashlib
    qhash = hashlib.md5(query.lower().encode()).hexdigest()
    c = _conn()
    row = c.execute("SELECT result, ts FROM jit_cache WHERE query_hash=?", (qhash,)).fetchone()
    if row and (time.time() - row[1]) < 3600:
        c.execute("UPDATE jit_cache SET hits=hits+1 WHERE query_hash=?", (qhash,))
        c.commit()
        return row[0]
    try:
        result = search_fn(query)
        if result and len(result) > 20:
            c.execute("""INSERT OR REPLACE INTO jit_cache
                (query_hash, query, result, source, confidence, ts)
                VALUES (?,?,?,?,?,?)""",
                (qhash, query[:200], result[:800], "jit_search", 0.8, time.time()))
            c.commit()
            return result
    except Exception as e:
        print(f"[JIT] fetch error: {e}")
    return None

def build_jit_context(msg: str, search_fn, complexity: str = "medium") -> str:
    """
    Pre-generation: detect what facts will likely be needed and fetch them.
    Returns context string to inject into system prompt.
    """
    if complexity == "easy":
        return ""
    gaps = detect_knowledge_gaps("", msg)
    if not gaps:
        return ""
    parts = []
    for gap_type, query in gaps[:2]:
        result = jit_fetch(query, search_fn)
        if result:
            parts.append(f"[JIT:{gap_type}] {query[:60]}: {result[:200]}")
    if not parts:
        return ""
    return "\n<jit_retrieved>\n" + "\n".join(parts) + "\n</jit_retrieved>"

_jit_instance = None

def get_jit():
    global _jit_instance
    if _jit_instance is None:
        _jit_instance = {"cache": {}, "lock": threading.Lock()}
    return _jit_instance
