from __future__ import annotations
import sqlite3, time, threading, re
from pathlib import Path

DB_PATH = Path.home() / "eliteomni_memory_v2.db"
_LOCK   = threading.Lock()

def _con():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-16000")
    return con

def init_db():
    with _LOCK:
        con = _con()
        con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
            USING fts5(text, source, tokenize='porter ascii')""")
        con.execute("""CREATE TABLE IF NOT EXISTS memory_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fts_rowid INTEGER, source TEXT DEFAULT 'conversation',
            ts REAL NOT NULL, skill TEXT DEFAULT 'general')""")
        con.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
            USING fts5(text, tokenize='porter ascii')""")
        con.execute("""CREATE TABLE IF NOT EXISTS episodic_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fts_rowid INTEGER, ts REAL NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS kv (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, ts REAL NOT NULL)""")
        con.commit(); con.close()

init_db()

def mem_save(text: str, source: str = "conversation", skill: str = "general"):
    if not text or len(text.strip()) < 5:
        return
    text = text[:1000].strip()
    with _LOCK:
        try:
            con = _con()
            cur = con.execute("INSERT INTO memory_fts(text,source) VALUES(?,?)", (text,source))
            con.execute("INSERT INTO memory_meta(fts_rowid,source,ts,skill) VALUES(?,?,?,?)",
                        (cur.lastrowid, source, time.time(), skill))
            con.execute("""DELETE FROM memory_meta WHERE id NOT IN (
                SELECT id FROM memory_meta ORDER BY ts DESC LIMIT 10000)""")
            con.commit(); con.close()
        except Exception as e:
            print(f"[MemFast] save error: {e}")

def mem_get(query: str, k: int = 6) -> list[str]:
    if not query or len(query.strip()) < 3:
        return []
    clean = re.sub(r"[^\w\s]", " ", query)
    words = [w for w in clean.split() if len(w) >= 3]
    if not words:
        return []
    fts_query = " OR ".join(words[:8])
    try:
        with _LOCK:
            con = _con()
            rows = con.execute(
                "SELECT text FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts_query, k)).fetchall()
            con.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[MemFast] get error: {e}")
        return []

def episodic_save(text: str):
    if not text:
        return
    with _LOCK:
        try:
            con = _con()
            cur = con.execute("INSERT INTO episodic_fts(text) VALUES(?)", (text[:500],))
            con.execute("INSERT INTO episodic_meta(fts_rowid,ts) VALUES(?,?)",
                        (cur.lastrowid, time.time()))
            con.execute("""DELETE FROM episodic_meta WHERE id NOT IN (
                SELECT id FROM episodic_meta ORDER BY ts DESC LIMIT 500)""")
            con.commit(); con.close()
        except Exception as e:
            print(f"[MemFast] episodic_save error: {e}")

def episodic_get(query: str, k: int = 3) -> list[str]:
    if not query:
        return []
    clean = re.sub(r"[^\w\s]", " ", query)
    words = [w for w in clean.split() if len(w) >= 3]
    if not words:
        return []
    fts_query = " OR ".join(words[:6])
    try:
        with _LOCK:
            con = _con()
            rows = con.execute(
                "SELECT text FROM episodic_fts WHERE episodic_fts MATCH ? LIMIT ?",
                (fts_query, k)).fetchall()
            con.close()
        return [r[0] for r in rows]
    except Exception as e:
        print(f"[MemFast] episodic_get error: {e}")
        return []

def kv_set(key: str, value: str):
    with _LOCK:
        try:
            con = _con()
            con.execute("INSERT OR REPLACE INTO kv(key,value,ts) VALUES(?,?,?)",
                        (key, value[:5000], time.time()))
            con.commit(); con.close()
        except Exception as e:
            print(f"[MemFast] kv_set error: {e}")

def kv_get(key: str) -> str:
    with _LOCK:
        try:
            con = _con()
            row = con.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            con.close()
            return row[0] if row else ""
        except Exception as e:
            print(f"[MemFast] kv_get error: {e}")
            return ""

def stats() -> dict:
    try:
        with _LOCK:
            con = _con()
            mc = con.execute("SELECT COUNT(*) FROM memory_meta").fetchone()[0]
            ec = con.execute("SELECT COUNT(*) FROM episodic_meta").fetchone()[0]
            kc = con.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
            con.close()
        return {"memory_entries":mc,"episodic_entries":ec,"kv_entries":kc,
                "db_path":str(DB_PATH),"engine":"FTS5-BM25"}
    except Exception as e:
        return {"error":str(e)}

# ── Goodfellow Ch.15: Representation Learning via dense embeddings ────────────
import threading as _embed_th
_EMBED_OK = False
_EMBED_MODEL = None
_np = None

def _load_embed_model():
    global _EMBED_MODEL, _EMBED_OK, _np
    try:
        from sentence_transformers import SentenceTransformer as _ST
        import numpy as np
        _np = np
        _EMBED_MODEL = _ST("all-MiniLM-L6-v2")
        _EMBED_OK = True
        print("[embed] all-MiniLM-L6-v2 loaded — semantic search active")
    except Exception as _e:
        _EMBED_OK = False
        print(f"[embed] fallback to BM25: {_e}")

_embed_th.Thread(target=_load_embed_model, daemon=True).start()

def embed_and_rank(query: str, candidates: list[str], top_k: int = 5) -> list[str]:
    """
    Goodfellow §15.1: learned representations outperform hand-crafted features.
    Encodes query + candidates into dense vectors, ranks by cosine similarity.
    Falls back to BM25 keyword ranking if model unavailable.
    """
    if not candidates:
        return []
    if not _EMBED_OK or _EMBED_MODEL is None:
        # BM25-style keyword fallback
        import re
        kws = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        scored = [(sum(1 for k in kws if k in c.lower()), c) for c in candidates]
        scored.sort(reverse=True)
        return [c for _, c in scored[:top_k]]
    try:
        all_texts = [query] + candidates
        vecs = _EMBED_MODEL.encode(all_texts, normalize_embeddings=True, batch_size=32)
        q_vec = vecs[0]
        c_vecs = vecs[1:]
        # Cosine similarity (Goodfellow §2.7): dot product of L2-normalized vectors
        scores = _np.dot(c_vecs, q_vec)
        top_idx = _np.argsort(scores)[::-1][:top_k]
        return [candidates[i] for i in top_idx]
    except Exception as e:
        print(f"[embed] ranking error: {e}")
        return candidates[:top_k]

def mem_get_semantic(query: str, k: int = 6) -> list[str]:
    """Drop-in replacement for mem_get() using semantic ranking."""
    # First get BM25 candidates (cheap), then re-rank semantically (Goodfellow §15)
    candidates = mem_get(query, k=k * 3)
    return embed_and_rank(query, candidates, top_k=k)
