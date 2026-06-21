"""
knowledge_rag.py — Production async RAG pipeline.

Architecture:
  - SQLite (WAL mode, pooled via aiosqlite): chunk text, source, timestamps,
    parent-child links. Async so it doesn't block the event loop under load.
  - Qdrant (async client): vector index, keyed by chunk id. Does nearest-
    neighbor search — the part SQLite cannot do at this scale.
  - Mistral: embeddings, batched, retried, rate-limited via a semaphore so
    concurrent ingestion doesn't blow through provider rate limits.

All public entry points are async. A sync wrapper (`run_sync`) is provided
for callers (e.g. simple scripts) that aren't in an event loop.

Observability: every retrieval/ingest call emits a `RagMetrics` record via
the `on_metrics` callback (defaults to structured logging). Wire this to
your metrics backend (Prometheus/Datadog/etc.) in production.
"""

import os
import json
import time
import math
import hashlib
import logging
import asyncio
import uuid
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Optional, Callable

import aiosqlite

from modules.rag_config import config

logger = logging.getLogger("rag")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)r}'
    ))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# ── METRICS ─────────────────────────────────────────────────────────────────

@dataclass
class RagMetrics:
    operation: str            # "ingest" | "retrieve" | "embed"
    duration_ms: float
    chunk_count: int = 0
    cache_hit: bool = False
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _default_metrics_sink(m: RagMetrics):
    logger.info(json.dumps({
        "metric": "rag",
        "operation": m.operation,
        "duration_ms": round(m.duration_ms, 2),
        "chunk_count": m.chunk_count,
        "error": m.error,
        **m.extra,
    }))


_metrics_sink: Callable[[RagMetrics], None] = _default_metrics_sink


def set_metrics_sink(fn: Callable[[RagMetrics], None]):
    """Wire metrics to Prometheus/Datadog/etc. by replacing the sink."""
    global _metrics_sink
    _metrics_sink = fn


@asynccontextmanager
async def _timed(operation, **extra_on_success):
    start = time.monotonic()
    err = None
    try:
        yield
    except Exception as e:
        err = str(e)
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        _metrics_sink(RagMetrics(
            operation=operation,
            duration_ms=duration_ms,
            error=err,
            extra=extra_on_success,
        ))


# ── SQLITE (async, WAL, pooled) ────────────────────────────────────────────

_db_lock = asyncio.Lock()
_db_initialized = False


@asynccontextmanager
async def _conn():
    con = await aiosqlite.connect(config.memory_db, timeout=30)
    try:
        await con.execute("PRAGMA journal_mode=WAL")
        await con.execute("PRAGMA busy_timeout=5000")
        yield con
        await con.commit()
    finally:
        await con.close()


async def init_db():
    global _db_initialized
    async with _db_lock:
        if _db_initialized:
            return
        async with _conn() as con:
            await con.execute("""CREATE TABLE IF NOT EXISTS rag_docs (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                chunk       TEXT NOT NULL,
                source      TEXT,
                ts          REAL,
                parent_idx  INTEGER
            )""")
            await con.execute("""CREATE TABLE IF NOT EXISTS rag_parents (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                parent_idx  INTEGER,
                chunk       TEXT NOT NULL,
                source      TEXT,
                ts          REAL
            )""")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_rag_docs_doc_id ON rag_docs(doc_id)")
            await con.execute("CREATE INDEX IF NOT EXISTS idx_rag_parents_doc_id ON rag_parents(doc_id)")
        _db_initialized = True


# ── CHUNKER ──────────────────────────────────────────────────────────────────

def chunk_text(text, size=400, overlap=50):
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    if not text or not text.strip():
        return []
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def _chunk_id(doc_id, chunk):
    return hashlib.md5((doc_id + chunk[:80]).encode()).hexdigest()


def _id_to_uuid(hex_str):
    return str(uuid.UUID(hex_str[:32].ljust(32, "0")))


# ── EMBEDDINGS (async, batched, retried, rate-limited) ─────────────────────

_embed_semaphore = asyncio.Semaphore(config.max_concurrent_embed_requests)


async def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    async with _timed("embed", text_count=len(texts)):
        if not config.mistral_api_key:
            logger.warning("MISTRAL_API_KEY not set — using fallback vectorizer")
            return _fallback_embed(texts)

        out = []
        batches = [texts[i:i + config.embed_batch_size]
                   for i in range(0, len(texts), config.embed_batch_size)]
        results = await asyncio.gather(*[_embed_batch(b) for b in batches])
        for r in results:
            out.extend(r)
        return out


async def _embed_batch(batch: list[str]) -> list[list[float]]:
    import httpx
    async with _embed_semaphore:
        for attempt in range(config.embed_max_retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(
                        "https://api.mistral.ai/v1/embeddings",
                        headers={
                            "Authorization": "Bearer " + config.mistral_api_key,
                            "Content-Type": "application/json",
                        },
                        json={"model": config.embed_model, "input": batch},
                    )
                if r.status_code == 200:
                    data = r.json()["data"]
                    return [d["embedding"] for d in data]
                elif r.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"rate limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"embed http {r.status_code}: {r.text[:200]}")
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"embed request failed (attempt {attempt + 1}/{config.embed_max_retries}): {e}")
                await asyncio.sleep(2 ** attempt)
        logger.error(f"embedding batch of {len(batch)} failed after {config.embed_max_retries} retries — using fallback")
        return _fallback_embed(batch)


def _fallback_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic local hashing vectorizer — keyword-ish matching used when
    the embedding provider is unavailable. Keeps the system functional
    instead of hard-failing; quality is lower than real embeddings."""
    dim = config.embed_dim_fallback
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for w in t.lower().split():
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % dim
            v[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vecs.append([x / norm for x in v])
    return vecs


def cosine(a, b):
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


# ── QDRANT (async) ────────────────────────────────────────────────────────────

_qdrant_client = None
_qdrant_lock = asyncio.Lock()


async def _qdrant():
    global _qdrant_client
    async with _qdrant_lock:
        if _qdrant_client is None:
            from qdrant_client import AsyncQdrantClient
            kwargs = {"url": config.qdrant_url, "timeout": config.qdrant_timeout_s}
            if config.qdrant_api_key:
                kwargs["api_key"] = config.qdrant_api_key
            _qdrant_client = AsyncQdrantClient(**kwargs)
        return _qdrant_client


async def _ensure_collection(dim):
    from qdrant_client.models import Distance, VectorParams
    client = await _qdrant()
    existing = [c.name for c in (await client.get_collections()).collections]
    if config.qdrant_collection not in existing:
        await client.create_collection(
            collection_name=config.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"created qdrant collection {config.qdrant_collection} (dim={dim})")


async def _upsert_vectors(ids, vectors, payloads):
    from qdrant_client.models import PointStruct
    if not vectors:
        return
    await _ensure_collection(len(vectors[0]))
    points = [
        PointStruct(id=_id_to_uuid(i), vector=v, payload=p)
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client = await _qdrant()
    await client.upsert(collection_name=config.qdrant_collection, points=points)


async def _search_vectors(query_vec, top_k):
    try:
        client = await _qdrant()
        result = await client.query_points(
            collection_name=config.qdrant_collection,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        hits = result.points
    except Exception as e:
        logger.error(f"qdrant search failed: {e}")
        return []
    return [(h.payload.get("chunk_id"), h.score) for h in hits]


# ── DEDUPLICATION ────────────────────────────────────────────────────────────

async def _chunk_exists(con, chunk_id):
    async with con.execute("SELECT 1 FROM rag_docs WHERE id=?", (chunk_id,)) as cur:
        return await cur.fetchone() is not None


# ── INGEST ────────────────────────────────────────────────────────────────────

async def ingest(text: str, source: str = "user") -> int:
    await init_db()
    async with _timed("ingest", source=source) as _:
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()
        chunks = chunk_text(text)
        if not chunks:
            return 0

        async with _conn() as con:
            new_chunks, new_ids = [], []
            for c in chunks:
                cid = _chunk_id(doc_id, c)
                if not await _chunk_exists(con, cid):
                    new_chunks.append(c)
                    new_ids.append(cid)

            if not new_chunks:
                logger.info(f"ingest: all {len(chunks)} chunks already present (doc_id={doc_id})")
                return 0

            embs = await embed(new_chunks)
            now = time.time()
            await con.executemany(
                "INSERT OR IGNORE INTO rag_docs (id,doc_id,chunk,source,ts,parent_idx) VALUES (?,?,?,?,?,NULL)",
                [(cid, doc_id, c, source, now) for cid, c in zip(new_ids, new_chunks)],
            )

        await _upsert_vectors(
            new_ids, embs,
            [{"chunk_id": cid, "doc_id": doc_id, "source": source} for cid in new_ids],
        )
        logger.info(f"ingested {len(new_chunks)} new chunks (doc_id={doc_id})")
        return len(new_chunks)


async def ingest_with_parents(text: str, source: str = "user") -> int:
    await init_db()
    async with _timed("ingest_with_parents", source=source):
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()
        parent_chunks = chunk_text(text, size=800, overlap=100)
        child_chunks = chunk_text(text, size=150, overlap=30)
        if not child_chunks:
            return 0

        child_to_parent = {}
        for ci, cc in enumerate(child_chunks):
            best_pi, best_overlap = 0, -1
            cc_words = set(cc.split())
            for pi, pc in enumerate(parent_chunks):
                overlap = len(cc_words & set(pc.split()))
                if overlap > best_overlap:
                    best_overlap, best_pi = overlap, pi
            child_to_parent[ci] = best_pi

        now = time.time()
        async with _conn() as con:
            for pi, pc in enumerate(parent_chunks):
                pid = _chunk_id(doc_id, "parent" + str(pi) + pc)
                await con.execute(
                    "INSERT OR IGNORE INTO rag_parents (id,doc_id,parent_idx,chunk,source,ts) VALUES (?,?,?,?,?,?)",
                    (pid, doc_id, pi, pc, source, now),
                )

            new_children, new_ids, new_parent_idx = [], [], []
            for ci, cc in enumerate(child_chunks):
                cid = _chunk_id(doc_id, cc)
                if not await _chunk_exists(con, cid):
                    new_children.append(cc)
                    new_ids.append(cid)
                    new_parent_idx.append(child_to_parent[ci])

            if not new_children:
                logger.info(f"ingest_with_parents: all {len(child_chunks)} child chunks already present")
                return 0

            embs = await embed(new_children)
            await con.executemany(
                "INSERT OR IGNORE INTO rag_docs (id,doc_id,chunk,source,ts,parent_idx) VALUES (?,?,?,?,?,?)",
                [(cid, doc_id, c, source, now, pidx)
                 for cid, c, pidx in zip(new_ids, new_children, new_parent_idx)],
            )

        await _upsert_vectors(
            new_ids, embs,
            [{"chunk_id": cid, "doc_id": doc_id, "source": source, "parent_idx": pidx}
             for cid, pidx in zip(new_ids, new_parent_idx)],
        )
        logger.info(f"ingested {len(new_children)} child chunks + {len(parent_chunks)} parents (doc_id={doc_id})")
        return len(new_children)


# ── RETRIEVE ──────────────────────────────────────────────────────────────────

def _bm25_scores(query, chunks):
    k1, b = 1.5, 0.75
    tokenized = [c.lower().split() for c in chunks]
    avgdl = sum(len(t) for t in tokenized) / max(len(tokenized), 1)
    q_terms = query.lower().split()

    df = {}
    for t in tokenized:
        for term in set(t):
            df[term] = df.get(term, 0) + 1
    n = len(tokenized)

    scores = []
    for t in tokenized:
        dl = len(t)
        score = 0.0
        for term in q_terms:
            f = t.count(term)
            if f == 0:
                continue
            idf = math.log(1 + (n - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avgdl, 1e-9)))
        scores.append(score)
    return scores


async def _rerank(query: str, candidates: list[str], top_k: int) -> list[str]:
    """LLM-based listwise reranking: shows the model the query + numbered
    candidates, asks it to return the best top_k indices in order. This is
    the single highest-leverage retrieval-quality improvement available —
    BM25/vector scores are proxies for relevance, this reads query and
    chunk together the way the final answer-generation step will.

    Falls back to returning candidates unchanged (original order) on any
    failure — reranking is a quality enhancement, never load-bearing.
    Costs one extra LLM call per retrieve() when enabled."""
    if len(candidates) <= 1:
        return candidates[:top_k]
    try:
        from modules.core.http_client import mistral_generate
        numbered = "\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(candidates))
        prompt = (
            f"Question: {query}\n\n"
            f"Candidate passages:\n{numbered}\n\n"
            f"Return the indices of the {min(top_k, len(candidates))} most relevant "
            f"passages, most relevant first, as a JSON array of integers only. "
            f"Example: [3, 0, 7]"
        )
        raw = mistral_generate([{"role": "user", "content": prompt}], max_tokens=100)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        indices = json.loads(raw.strip())
        if not isinstance(indices, list) or not indices:
            raise ValueError("rerank did not return a usable index list")
        result = []
        seen = set()
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                result.append(candidates[i])
                seen.add(i)
        if not result:
            raise ValueError("no valid indices in rerank response")
        return result[:top_k]
    except Exception as e:
        logger.warning(f"rerank failed ({e}), falling back to original order")
        return candidates[:top_k]


async def retrieve(query: str, top_k: Optional[int] = None, fetch_k: Optional[int] = None,
                    use_parents: bool = True, history=None, rerank: bool = False) -> list[str]:
    await init_db()
    top_k = top_k or config.default_top_k
    fetch_k = fetch_k or config.default_fetch_k
    # when reranking, fetch more candidates than top_k so the reranker has
    # real choices to work with, not just the top_k already-blended results
    blend_k = max(top_k * 3, fetch_k) if rerank else top_k

    async with _timed("retrieve", query_len=len(query)) as _:
        q_vec = (await embed([query]))[0]
        hits = await _search_vectors(q_vec, top_k=fetch_k)
        if not hits:
            return []

        chunk_ids = [cid for cid, _ in hits]
        vec_scores = {cid: score for cid, score in hits}

        placeholders = ",".join("?" * len(chunk_ids))
        async with _conn() as con:
            async with con.execute(
                f"SELECT id, chunk, doc_id, parent_idx FROM rag_docs WHERE id IN ({placeholders})",
                chunk_ids,
            ) as cur:
                rows = await cur.fetchall()
        row_by_id = {r[0]: r for r in rows}

        ordered = [row_by_id[cid] for cid in chunk_ids if cid in row_by_id]
        if not ordered:
            return []

        texts = [r[1] for r in ordered]
        bm25 = _bm25_scores(query, texts)
        max_bm25 = max(bm25) or 1.0
        max_vec = max(vec_scores.values()) or 1.0

        blended = []
        for row, bm25_s in zip(ordered, bm25):
            cid, chunk, doc_id, parent_idx = row
            v_s = vec_scores.get(cid, 0.0) / max_vec
            b_s = bm25_s / max_bm25
            blended.append((config.vector_weight * v_s + config.bm25_weight * b_s,
                             chunk, doc_id, parent_idx))

        blended.sort(key=lambda x: x[0], reverse=True)
        top = blended[:blend_k]

        if rerank:
            candidate_texts = [c for _, c, _, _ in top]
            reranked_texts = await _rerank(query, candidate_texts, top_k)
            text_to_row = {c: (s, c, d, p) for s, c, d, p in top}
            top = [text_to_row[t] for t in reranked_texts if t in text_to_row]
        else:
            top = top[:top_k]

        if not use_parents:
            return [c for _, c, _, _ in top]

        results, seen_parents = [], set()
        async with _conn() as con:
            for _, chunk, doc_id, parent_idx in top:
                if parent_idx is None:
                    results.append(chunk)
                    continue
                key = (doc_id, parent_idx)
                if key in seen_parents:
                    continue
                seen_parents.add(key)
                async with con.execute(
                    "SELECT chunk FROM rag_parents WHERE doc_id=? AND parent_idx=?",
                    (doc_id, parent_idx),
                ) as cur:
                    prow = await cur.fetchone()
                results.append(prow[0] if prow else chunk)
        return results


async def multi_query_retrieve(query: str, top_k: Optional[int] = None, history=None,
                                n_queries: int = 3) -> list[str]:
    top_k = top_k or config.default_top_k
    try:
        from modules.core.http_client import mistral_generate
        prompt = (
            f"Generate {n_queries} different search queries that would help answer "
            f"this question, covering different phrasings or angles. "
            f"Output a JSON array of strings only, nothing else.\n\nQuestion: {query}"
        )
        raw = mistral_generate([{"role": "user", "content": prompt}], max_tokens=200)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        queries = json.loads(raw.strip())
        if not isinstance(queries, list) or not queries:
            raise ValueError("model did not return a query list")
    except Exception as e:
        logger.warning(f"multi_query_retrieve: rewrite failed ({e}), falling back to single query")
        return await retrieve(query, top_k=top_k, history=history)

    queries = [query] + [q for q in queries if isinstance(q, str)]
    per_query_k = max(2, top_k // len(queries) + 1)
    results_per_query = await asyncio.gather(
        *[retrieve(q, top_k=per_query_k, history=history) for q in queries]
    )

    seen, merged = set(), []
    for chunks in results_per_query:
        for chunk in chunks:
            key = hashlib.md5(chunk[:100].encode()).hexdigest()
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged[:top_k]


# ── CONTEXT INJECTION ────────────────────────────────────────────────────────

async def inject_rag(msgs: list[dict], query: str, top_k: Optional[int] = None,
                      history=None, use_multi_query: bool = False, rerank: bool = False) -> list[dict]:
    if use_multi_query:
        chunks = await multi_query_retrieve(query, top_k=top_k, history=history)
    else:
        chunks = await retrieve(query, top_k=top_k, history=history, rerank=rerank)
    if not chunks:
        return msgs

    ctx = "RETRIEVED CONTEXT (use this to answer):\n" + "\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(chunks)
    )

    result, injected = [], False
    for m in msgs:
        if m.get("role") == "system" and not injected:
            result.append({**m, "content": m["content"] + "\n\n" + ctx})
            injected = True
        else:
            result.append(m)
    if not injected:
        result = [{"role": "system", "content": ctx}] + msgs

    logger.info(f"injected {len(chunks)} chunks")
    return result


async def rag_ask(query: str, msgs: Optional[list[dict]] = None, skill: str = "general",
                   max_tokens: int = 2000, top_k: Optional[int] = None,
                   use_multi_query: bool = False, rerank: bool = False) -> str:
    from modules.guardrails import gateway
    base = (msgs or []) + [{"role": "user", "content": query}]
    rag_msgs = await inject_rag(base, query, top_k=top_k, history=msgs,
                                 use_multi_query=use_multi_query, rerank=rerank)
    result = gateway(rag_msgs, skill=skill, max_tokens=max_tokens)
    return result["response"] if not asyncio.iscoroutine(result) else (await result)["response"]


# ── SYNC WRAPPER (for non-async callers) ────────────────────────────────────

def run_sync(coro):
    """Run an async RAG call from synchronous code. Do not call this from
    inside an already-running event loop (e.g. inside FastAPI handlers) —
    await the async function directly there instead."""
    return asyncio.run(coro)
