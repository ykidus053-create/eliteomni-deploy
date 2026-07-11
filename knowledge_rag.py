"""Compatibility exports for the canonical modules.knowledge_rag implementation."""
from modules.knowledge_rag import (
    BOOK_MODULES,
    build_index,
    chunk_text,
    cosine,
    get_knowledge_context,
    ingest,
    ingest_with_parents,
    init_db,
    inject_rag,
    retrieve,
    start_background_indexer,
)

__all__ = [
    "BOOK_MODULES",
    "build_index",
    "chunk_text",
    "cosine",
    "get_knowledge_context",
    "ingest",
    "ingest_with_parents",
    "init_db",
    "inject_rag",
    "retrieve",
    "start_background_indexer",
]
