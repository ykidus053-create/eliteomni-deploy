"""Async local hybrid RAG with durable SQLite storage.

This module provides the API exercised by EliteOmni's RAG tests while keeping
compatibility helpers used by the existing application. It does not claim that
SQLite plus deterministic local embeddings is a distributed vector database.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from modules.rag_config import config

_DB = os.path.expanduser(config.memory_db)
_qdrant_client: Any = None
_cache: dict[str, list[dict[str, Any]]] = {}
_cache_lock = threading.Lock()

BOOK_MODULES = [
    "book_gaps_impl",
    "book8_gaps",
    "final_gaps",
    "aie_book_impl",
    "dl_book_implementations",
    "dl_book_implementations2",
    "dl_book_implementations3",
    "goodfellow_dl",
    "gaps_all_books",
    "math_impl",
]

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_FALLBACK_DIM = 256


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def chunk_text(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping word chunks."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = (text or "").split()
    if not words:
        return []

    step = size - overlap
    return [
        " ".join(words[start : start + size])
        for start in range(0, len(words), step)
        if words[start : start + size]
    ]


def _fallback_embed(texts: Sequence[str]) -> list[list[float]]:
    """Create deterministic normalized sparse-hash embeddings."""
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * _FALLBACK_DIM
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % _FALLBACK_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        vectors.append(vector)
    return vectors


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity, or zero for invalid/zero vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _bm25_scores(query: str, chunks: Sequence[str]) -> list[float]:
    """Compute BM25 scores across a candidate collection."""
    documents = [_tokens(chunk) for chunk in chunks]
    if not documents:
        return []

    query_terms = _tokens(query)
    if not query_terms:
        return [0.0] * len(documents)

    average_length = sum(len(doc) for doc in documents) / max(1, len(documents))
    if average_length == 0:
        return [0.0] * len(documents)

    document_frequency = {
        term: sum(1 for doc in documents if term in doc)
        for term in set(query_terms)
    }
    k1 = 1.5
    b = 0.75
    scores: list[float] = []

    for document in documents:
        score = 0.0
        for term in query_terms:
            frequency = document.count(term)
            if frequency == 0:
                continue
            df = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(
                1.0 + (len(documents) - df + 0.5) / (df + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * len(document) / average_length
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1.0)
            ) / denominator
        scores.append(score)
    return scores


def _connect() -> sqlite3.Connection:
    path = Path(_DB).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(path),
        timeout=30,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


class _AsyncCursor:
    def __init__(
        self,
        connection: sqlite3.Connection,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> None:
        self._connection = connection
        self._sql = sql
        self._parameters = tuple(parameters)
        self._cursor: sqlite3.Cursor | None = None

    async def _ensure(self) -> "_AsyncCursor":
        if self._cursor is None:
            self._cursor = await asyncio.to_thread(
                self._connection.execute,
                self._sql,
                self._parameters,
            )
        return self

    def __await__(self):
        return self._ensure().__await__()

    async def __aenter__(self) -> "_AsyncCursor":
        return await self._ensure()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._cursor is not None:
            await asyncio.to_thread(self._cursor.close)

    async def fetchone(self):
        await self._ensure()
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self):
        await self._ensure()
        return await asyncio.to_thread(self._cursor.fetchall)


class _AsyncConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
    ) -> _AsyncCursor:
        return _AsyncCursor(self._connection, sql, parameters)

    async def commit(self) -> None:
        await asyncio.to_thread(self._connection.commit)

    async def rollback(self) -> None:
        await asyncio.to_thread(self._connection.rollback)

    async def close(self) -> None:
        await asyncio.to_thread(self._connection.close)


@asynccontextmanager
async def _conn():
    """Yield a minimal aiosqlite-compatible connection wrapper."""
    connection = await asyncio.to_thread(_connect)
    wrapper = _AsyncConnection(connection)
    try:
        yield wrapper
        await wrapper.commit()
    except Exception:
        await wrapper.rollback()
        raise
    finally:
        await wrapper.close()


async def init_db() -> None:
    """Create the durable local RAG schema."""

    def initialize() -> None:
        with _connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_parents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    parent_id TEXT,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(parent_id) REFERENCES rag_parents(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_rag_chunks_source
                    ON rag_chunks(source);
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_parent
                    ON rag_chunks(parent_id);
                """
            )

    await asyncio.to_thread(initialize)


def _chunk_id(source: str, text: str) -> str:
    payload = f"{source}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def ingest(
    text: str,
    source: str = "unknown",
    *,
    chunk_size: int = 400,
    overlap: int = 50,
) -> int:
    """Insert new chunks and return the number actually added."""
    chunks = chunk_text(text, size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    await init_db()
    embeddings = _fallback_embed(chunks)
    now = time.time()

    def insert_rows() -> int:
        with _connect() as connection:
            before = connection.total_changes
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO rag_chunks(
                        id, source, parent_id, chunk_index,
                        text, embedding, created_at
                    ) VALUES (?, ?, NULL, ?, ?, ?, ?)
                    """,
                    (
                        _chunk_id(source, chunk),
                        source,
                        index,
                        chunk,
                        json.dumps(embedding),
                        now,
                    ),
                )
            return connection.total_changes - before

    inserted = await asyncio.to_thread(insert_rows)
    if inserted:
        with _cache_lock:
            _cache.clear()
    return inserted


async def ingest_with_parents(
    text: str,
    source: str = "unknown",
    *,
    parent_size: int = 1200,
    child_size: int = 400,
    overlap: int = 50,
) -> int:
    """Store parent chunks and linked retrieval children atomically."""
    parents = chunk_text(text, size=parent_size, overlap=min(overlap, parent_size - 1))
    if not parents:
        return 0

    await init_db()
    now = time.time()

    records: list[tuple[str, str, int, str, list[float]]] = []
    for parent_index, parent in enumerate(parents):
        parent_id = _chunk_id(f"{source}:parent:{parent_index}", parent)
        children = chunk_text(
            parent,
            size=child_size,
            overlap=min(overlap, child_size - 1),
        )
        embeddings = _fallback_embed(children)
        for child_index, (child, embedding) in enumerate(
            zip(children, embeddings)
        ):
            records.append(
                (parent_id, child, child_index, parent, embedding)
            )

    def insert_tree() -> int:
        with _connect() as connection:
            before = connection.total_changes
            for parent_id, child, child_index, parent, embedding in records:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO rag_parents(
                        id, source, text, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (parent_id, source, parent, now),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO rag_chunks(
                        id, source, parent_id, chunk_index,
                        text, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _chunk_id(f"{source}:{parent_id}", child),
                        source,
                        parent_id,
                        child_index,
                        child,
                        json.dumps(embedding),
                        now,
                    ),
                )
            return connection.total_changes - before

    changes = await asyncio.to_thread(insert_tree)
    if changes:
        with _cache_lock:
            _cache.clear()

    # Count only child rows, not newly inserted parent rows.
    return min(len(records), changes)


async def _rerank(
    query: str,
    candidates: Sequence[Any],
    top_k: int = 5,
) -> list[Any]:
    """Return bounded candidates; preserve order when no reranker is configured."""
    del query
    if top_k <= 0:
        return []
    return list(candidates[:top_k])


async def retrieve(
    query: str,
    top_k: int | None = None,
    min_score: float = 0.0,
    rerank: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve local chunks using deterministic hybrid lexical/vector scoring."""
    query = (query or "").strip()
    if not query:
        return []

    limit = top_k or config.default_top_k
    if limit <= 0:
        return []

    await init_db()
    cache_key = f"{query}\0{limit}\0{min_score}\0{rerank}"
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

    def load_rows():
        with _connect() as connection:
            return connection.execute(
                """
                SELECT id, source, parent_id, text, embedding
                FROM rag_chunks
                ORDER BY created_at, chunk_index
                """
            ).fetchall()

    rows = await asyncio.to_thread(load_rows)
    if not rows:
        return []

    texts = [row["text"] for row in rows]
    lexical = _bm25_scores(query, texts)
    query_embedding = _fallback_embed([query])[0]

    results: list[dict[str, Any]] = []
    for row, lexical_score in zip(rows, lexical):
        try:
            embedding = json.loads(row["embedding"])
        except (TypeError, json.JSONDecodeError):
            embedding = _fallback_embed([row["text"]])[0]

        vector_score = max(0.0, cosine(query_embedding, embedding))
        normalized_bm25 = lexical_score / (1.0 + lexical_score)
        score = (
            config.vector_weight * vector_score
            + config.bm25_weight * normalized_bm25
        )
        if score >= min_score:
            results.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "parent_id": row["parent_id"],
                    "text": row["text"],
                    "score": score,
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)
    bounded: list[dict[str, Any]] = results[: max(limit, 1)]
    if rerank:
        bounded = await _rerank(query, bounded, top_k=limit)

    with _cache_lock:
        if len(_cache) >= 500:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = [dict(item) for item in bounded]

    return bounded


async def inject_rag(
    messages: Sequence[dict[str, Any]],
    query: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Inject retrieved passages into the system message without mutating input."""
    hits = await retrieve(query, top_k=top_k)
    if not hits:
        return list(messages)

    context = "\n\n".join(
        f"[{hit['source']}] {hit['text']}" for hit in hits
    )
    block = (
        "\n\n[RETRIEVED CONTEXT]\n"
        + context
        + "\n[END RETRIEVED CONTEXT]"
    )

    output = [dict(message) for message in messages]
    for message in output:
        if message.get("role") == "system":
            message["content"] = str(message.get("content", "")) + block
            return output

    return [{"role": "system", "content": block.strip()}, *output]


def _run_sync(coroutine):
    """Run an async operation from synchronous application code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[Any] = []
    error: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:  # preserve the original failure
            error.append(exc)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


def get_knowledge_context(
    query: str,
    top_k: int = 8,
    max_tokens: int = 1500,
) -> str:
    """Synchronous compatibility wrapper used by the existing app."""
    hits = _run_sync(retrieve(query, top_k=top_k))
    if not hits:
        return ""

    maximum_characters = max(1, max_tokens) * 4
    lines = ["[RELEVANT KNOWLEDGE]"]
    for hit in hits:
        line = f"- [{hit['source']}] {hit['text']}"
        candidate = "\n".join([*lines, line, "[END KNOWLEDGE]"])
        if len(candidate) > maximum_characters:
            break
        lines.append(line)
    lines.append("[END KNOWLEDGE]")
    return "\n".join(lines)


def _extract_chunks(module_name: str) -> list[dict[str, str]]:
    """Extract documented public callables from a Python module."""
    chunks: list[dict[str, str]] = []
    try:
        module = (
            sys.modules[module_name]
            if module_name in sys.modules
            else importlib.import_module(module_name)
        )
        for name in dir(module):
            if name.startswith("_"):
                continue
            obj = getattr(module, name, None)
            if not callable(obj):
                continue
            doc = (inspect.getdoc(obj) or "").strip()
            try:
                signature = str(inspect.signature(obj))
            except (TypeError, ValueError):
                signature = ""
            chunks.append(
                {
                    "module": module_name,
                    "name": name,
                    "kind": "class" if isinstance(obj, type) else "function",
                    "doc": doc[:500],
                    "signature": signature[:200],
                    "chunk": f"{name}{signature}: {doc}",
                }
            )
    except Exception as exc:
        print(f"[knowledge_rag] extract error {module_name}: {exc}")
    return chunks


def build_index(force: bool = False) -> int:
    """Index documented book helpers into the local RAG store."""
    async def build() -> int:
        await init_db()

        if force:
            def clear_book_rows() -> None:
                with _connect() as connection:
                    connection.execute(
                        "DELETE FROM rag_chunks WHERE source LIKE 'book:%'"
                    )

            await asyncio.to_thread(clear_book_rows)

        inserted = 0
        for module_name in BOOK_MODULES:
            extracted = _extract_chunks(module_name)
            text = "\n\n".join(item["chunk"] for item in extracted)
            if text:
                inserted += await ingest(
                    text,
                    source=f"book:{module_name}",
                    chunk_size=250,
                    overlap=25,
                )
        return inserted

    return _run_sync(build())


def _init_db() -> None:
    """Legacy synchronous alias."""
    _run_sync(init_db())


def start_background_indexer(interval_seconds: int = 1800) -> threading.Thread:
    """Start one daemon index refresh loop and return its thread."""
    interval = max(60, int(interval_seconds))

    def worker() -> None:
        time.sleep(5)
        while True:
            try:
                build_index(force=True)
            except Exception as exc:
                print(f"[knowledge_rag] background index error: {exc}")
            time.sleep(interval)

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name="knowledge_indexer",
    )
    thread.start()
    return thread
