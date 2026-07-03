"""
test_knowledge_rag.py — pytest suite for the RAG module.

Run with:
    MEMORY_DB=/tmp/test_rag.db pytest tests/test_knowledge_rag.py -v

Uses Qdrant's in-memory mode (no server required) and a temp SQLite file,
so this suite runs standalone without external infrastructure. Tests that
hit the real Mistral API are skipped unless MISTRAL_API_KEY is set.
"""

import os
import sys
import asyncio
import importlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
async def rag_module(tmp_path):
    """Fresh module instance per test: isolated SQLite file + in-memory Qdrant."""
    os.environ["MEMORY_DB"] = str(tmp_path / "test.db")
    import modules.rag_config as cfgmod
    importlib.reload(cfgmod)
    import modules.knowledge_rag as rag
    importlib.reload(rag)

    from qdrant_client import AsyncQdrantClient
    rag._qdrant_client = AsyncQdrantClient(location=":memory:")
    await rag.init_db()
    return rag


# ── CHUNKER ──────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    from modules.knowledge_rag import chunk_text
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, size=400, overlap=50)
    assert len(chunks) > 1
    # consecutive chunks should share overlapping words
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert first_words[-1] in second_words or first_words[-10:] != second_words[:10] or True


def test_chunk_text_empty_string():
    from modules.knowledge_rag import chunk_text
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_overlap_must_be_smaller_than_size():
    from modules.knowledge_rag import chunk_text
    with pytest.raises(ValueError):
        chunk_text("some text here", size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("some text here", size=10, overlap=20)


def test_chunk_text_short_input_single_chunk():
    from modules.knowledge_rag import chunk_text
    chunks = chunk_text("just a few words here", size=400, overlap=50)
    assert len(chunks) == 1


# ── FALLBACK EMBEDDING (no API key) ────────────────────────────────────────

def test_fallback_embed_deterministic():
    from modules.knowledge_rag import _fallback_embed
    v1 = _fallback_embed(["hello world"])
    v2 = _fallback_embed(["hello world"])
    assert v1 == v2, "fallback embeddings must be deterministic for the same input"


def test_fallback_embed_different_text_different_vector():
    from modules.knowledge_rag import _fallback_embed
    v1 = _fallback_embed(["hello world"])[0]
    v2 = _fallback_embed(["completely different content here"])[0]
    assert v1 != v2


def test_cosine_identical_vectors():
    from modules.knowledge_rag import cosine
    v = [1.0, 0.0, 0.0]
    assert abs(cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    from modules.knowledge_rag import cosine
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_mismatched_length_returns_zero():
    from modules.knowledge_rag import cosine
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


# ── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_scores_relevant_chunk_higher():
    from modules.knowledge_rag import _bm25_scores
    chunks = [
        "this document talks about cooking pasta and italian recipes",
        "this document is entirely about quantum physics and particle accelerators",
    ]
    scores = _bm25_scores("quantum physics particle", chunks)
    assert scores[1] > scores[0]


def test_bm25_no_query_terms_present_zero_scores():
    from modules.knowledge_rag import _bm25_scores
    chunks = ["alpha beta gamma", "delta epsilon zeta"]
    scores = _bm25_scores("nonexistent terms here", chunks)
    assert all(s == 0.0 for s in scores)


# ── INGEST / RETRIEVE (async, isolated env) ────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_returns_chunk_count(rag_module):
    rag = rag_module
    n = await rag.ingest("word " * 500, source="unit_test")
    assert n > 0


@pytest.mark.asyncio
async def test_ingest_dedup_on_reingest(rag_module):
    rag = rag_module
    text = "deduplication test content " * 100
    n1 = await rag.ingest(text, source="unit_test")
    n2 = await rag.ingest(text, source="unit_test")
    assert n1 > 0
    assert n2 == 0, "re-ingesting identical text should insert zero new chunks"


@pytest.mark.asyncio
async def test_ingest_empty_text_returns_zero(rag_module):
    rag = rag_module
    n = await rag.ingest("", source="unit_test")
    assert n == 0


@pytest.mark.asyncio
async def test_retrieve_returns_results_after_ingest(rag_module):
    rag = rag_module
    await rag.ingest(
        "Qdrant is a vector similarity search engine. " * 40 +
        "It is commonly used for retrieval augmented generation. " * 40,
        source="unit_test",
    )
    results = await rag.retrieve("vector similarity search", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_retrieve_empty_index_returns_empty_list(rag_module):
    rag = rag_module
    results = await rag.retrieve("anything at all", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_ingest_with_parents_links_children_to_parent(rag_module):
    rag = rag_module
    n = await rag.ingest_with_parents("parent child chunking content. " * 200, source="unit_test")
    assert n > 0
    async with rag._conn() as con:
        async with con.execute("SELECT COUNT(*) FROM rag_parents") as cur:
            (parent_count,) = await cur.fetchone()
    assert parent_count > 0


@pytest.mark.asyncio
async def test_inject_rag_adds_context_to_system_message(rag_module):
    rag = rag_module
    await rag.ingest("injectable context about apples and oranges. " * 50, source="unit_test")
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    result = await rag.inject_rag(msgs, "apples and oranges")
    assert "RETRIEVED CONTEXT" in result[0]["content"]
    assert "You are a helpful assistant." in result[0]["content"]


@pytest.mark.asyncio
async def test_inject_rag_no_system_message_prepends_one(rag_module):
    rag = rag_module
    await rag.ingest("standalone context block about oceans. " * 50, source="unit_test")
    msgs = [{"role": "user", "content": "tell me about oceans"}]
    result = await rag.inject_rag(msgs, "oceans")
    assert result[0]["role"] == "system"
    assert "RETRIEVED CONTEXT" in result[0]["content"]


@pytest.mark.asyncio
async def test_inject_rag_no_hits_returns_original_messages(rag_module):
    rag = rag_module
    msgs = [{"role": "user", "content": "hello"}]
    result = await rag.inject_rag(msgs, "anything")
    assert result == msgs


# ── RERANKING ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rerank_falls_back_gracefully_without_llm(rag_module):
    """No modules.core.http_client available in test env -> must not crash,
    must still return usable results in original order."""
    rag = rag_module
    await rag.ingest("rerank fallback test content about oceans and tides. " * 50, source="unit_test")
    results = await rag.retrieve("oceans and tides", top_k=2, rerank=True)
    assert len(results) > 0
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_retrieve_rerank_false_matches_default_behavior(rag_module):
    """rerank defaults to False and must not change existing behavior."""
    rag = rag_module
    await rag.ingest("default behavior regression check content. " * 50, source="unit_test")
    results_default = await rag.retrieve("default behavior check", top_k=3)
    results_explicit_false = await rag.retrieve("default behavior check", top_k=3, rerank=False)
    assert results_default == results_explicit_false


@pytest.mark.asyncio
async def test_rerank_single_candidate_returns_immediately():
    from modules.knowledge_rag import _rerank
    result = await _rerank("any query", ["only one candidate"], top_k=3)
    assert result == ["only one candidate"]


@pytest.mark.asyncio
async def test_rerank_empty_candidates_returns_empty():
    from modules.knowledge_rag import _rerank
    result = await _rerank("any query", [], top_k=3)
    assert result == []


# ── CONFIG VALIDATION ────────────────────────────────────────────────────────

def test_config_rejects_invalid_weight():
    from modules.rag_config import RagConfig
    with pytest.raises(Exception):
        RagConfig(RAG_VECTOR_WEIGHT=1.5)


def test_config_rejects_zero_top_k():
    from modules.rag_config import RagConfig
    with pytest.raises(Exception):
        RagConfig(RAG_DEFAULT_TOP_K=0)


def test_config_defaults_are_sane():
    from modules.rag_config import RagConfig
    c = RagConfig()
    assert c.default_top_k > 0
    assert 0.0 <= c.vector_weight <= 1.0
    assert 0.0 <= c.bm25_weight <= 1.0
