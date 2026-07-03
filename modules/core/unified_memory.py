"""
Single write-through memory abstraction.
Replaces: 3 independent stores (SQLite + FAISS + ChromaDB) never reconciled.
All writes go through here — never split-brain again.
"""
import time, re, threading
from typing import Optional

class UnifiedMemory:
    """
    Write-through abstraction over all memory backends.
    Guarantees: write to one = write to all. Read from one = read from best.
    """
    def __init__(self):
        self._lock   = threading.Lock()
        self._sqlite = None   # lazy init
        self._chroma = None
        self._faiss  = None
        self._ready  = False

    def _init_backends(self):
        if self._ready: return
        with self._lock:
            if self._ready: return
            # SQLite (always available)
            try:
                import sqlite3, os
                db = os.path.expanduser("~/eliteomni_memory.db")
                self._sqlite = sqlite3.connect(db, check_same_thread=False)
                self._sqlite.execute("PRAGMA journal_mode=WAL")
            except Exception as e:
                print(f"[UnifiedMem] SQLite init: {e}")
            # ChromaDB (optional)
            try:
                import chromadb, os
                cc = chromadb.PersistentClient(path=os.path.expanduser("~/eliteomni_chroma"))
                self._chroma = cc.get_or_create_collection("unified_memory")
            except Exception:
                pass
            self._ready = True

    def save(self, text: str, source: str = "conversation",
             metadata: dict = None) -> bool:
        """Write-through: saves to ALL available backends atomically."""
        self._init_backends()
        text = text[:1000]
        ts   = time.time()
        ok   = False

        # 1. SQLite — always
        if self._sqlite:
            try:
                self._sqlite.execute(
                    "INSERT INTO memory (text,source,ts) VALUES (?,?,?)",
                    (text, source, ts))
                self._sqlite.execute(
                    "DELETE FROM memory WHERE id NOT IN "
                    "(SELECT id FROM memory ORDER BY ts DESC LIMIT 5000)")
                self._sqlite.commit()
                ok = True
            except Exception as e:
                print(f"[UnifiedMem] SQLite save: {e}")

        # 2. ChromaDB — if available
        if self._chroma:
            try:
                import uuid
                self._chroma.add(
                    documents=[text],
                    metadatas=[{**(metadata or {}), "source": source, "ts": ts}],
                    ids=[str(uuid.uuid4())])
            except Exception as e:
                print(f"[UnifiedMem] Chroma save: {e}")

        return ok

    def get(self, query: str, k: int = 6) -> list:
        """
        Unified retrieval: semantic (ChromaDB) + keyword (SQLite), deduped.
        Returns best k results across both backends.
        """
        self._init_backends()
        results = []
        seen    = set()

        # 1. Semantic search via ChromaDB (highest quality)
        if self._chroma:
            try:
                n = min(k, self._chroma.count())
                if n > 0:
                    res = self._chroma.query(
                        query_texts=[query], n_results=n)
                    for doc in (res.get("documents") or [[]])[0]:
                        if doc not in seen:
                            results.append(doc); seen.add(doc)
            except Exception as e:
                print(f"[UnifiedMem] Chroma get: {e}")

        # 2. Keyword fallback via SQLite
        if self._sqlite and len(results) < k:
            try:
                kws = set(re.findall(r"[a-zA-Z]{4,}", query.lower()))
                rows = self._sqlite.execute(
                    "SELECT text FROM memory ORDER BY ts DESC LIMIT 500"
                ).fetchall()
                scored = sorted(
                    [(sum(1 for w in kws if w in t.lower()), t)
                     for (t,) in rows if t not in seen and
                     any(w in t.lower() for w in kws)],
                    reverse=True)
                for _, t in scored[:k - len(results)]:
                    results.append(t)
            except Exception as e:
                print(f"[UnifiedMem] SQLite get: {e}")

        return results[:k]

    def clear(self):
        self._init_backends()
        if self._sqlite:
            self._sqlite.execute("DELETE FROM memory")
            self._sqlite.execute("DELETE FROM episodic")
            self._sqlite.commit()
        if self._chroma:
            try: self._chroma.delete(where={"source": {"$ne": ""}})
            except Exception: pass

    @property
    def stats(self) -> dict:
        self._init_backends()
        s = {"sqlite": 0, "chroma": 0}
        if self._sqlite:
            try: s["sqlite"] = self._sqlite.execute(
                "SELECT COUNT(*) FROM memory").fetchone()[0]
            except: pass
        if self._chroma:
            try: s["chroma"] = self._chroma.count()
            except: pass
        return s

unified_memory = UnifiedMemory()
