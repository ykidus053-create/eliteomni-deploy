
# Claude-style memory honesty
MEMORY_HONESTY = """
When using memory:
- Only surface memories that are clearly relevant to the current question
- Never confabulate memories that were not actually stored
- If a memory seems outdated or contradicted by the current conversation, flag it
- Prefer recent memories over older ones for factual claims
"""
# AUTO-SPLIT FROM app.py lines 2837-2878
import os, re, time, math, json, asyncio, random, ast, subprocess, sys, tempfile
from threading import Lock
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import urllib.request, urllib.parse

_chroma_client = None
_chroma_col    = None
_embedder      = None

def _init_semantic_memory():
    global _chroma_client, _chroma_col, _embedder
    try:
        import chromadb
        from fastembed import TextEmbedding
        _chroma_client = chromadb.PersistentClient(path=os.path.expanduser("~/eliteomni_chroma"))
        _chroma_col    = _chroma_client.get_or_create_collection("memory")
        _embedder      = TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir="/home/kidus/.fastembed_cache")
        print("[SemanticMem] Ready — chromadb + fastembed bge-small")
    except Exception as e:
        print(f"[SemanticMem] Not available: {e} — pip install chromadb sentence-transformers")

import threading as _sm_th
_sm_th.Thread(target=_init_semantic_memory, daemon=True).start()

def semantic_mem_save(text: str, meta: dict = None):
    """Save text to vector store."""
    if not _chroma_col or not _embedder: return
    try:
        import uuid
        emb = list(_embedder.embed([text]))[0].tolist()
        _chroma_col.add(embeddings=emb, documents=[text],
                        metadatas=[meta or {}], ids=[str(uuid.uuid4())])
    except Exception as e:
        print(f"[SemanticMem save] {e}")

def semantic_mem_get(query: str, k: int = 6) -> list:
    """Retrieve semantically similar memories."""
    if not _chroma_col or not _embedder: return []
    try:
        emb = list(_embedder.embed([query]))[0].tolist()
        results = _chroma_col.query(query_embeddings=emb, n_results=min(k, _chroma_col.count()), include=["documents","distances"])
        docs = results["documents"][0] if results["documents"] else []
        dists = results.get("distances", [[]])[0] if results.get("distances") else [0.0]*len(docs)
        hits = [d for d, s in zip(docs, dists) if s < 0.5]
        # Fei-Fei: reinforce retrieved memories (Hebbian — used = stronger)
        try:
            from modules.services.memory import mem_increment_hit
            for h in hits: mem_increment_hit(h)
        except Exception: pass
        return hits
    except Exception as e:
        print(f"[SemanticMem get] {e}")
        return []

# ── FINE-TUNE DATA COLLECTOR ─────────────────────────────────────────────────
