"""Vector store implementations for semantic search.

This module provides:
- VectorStore: Protocol for vector storage backends
- SearchResult: Result model for vector search
- InMemoryVectorStore: Simple in-memory implementation
- QdrantVectorStore: Qdrant-backed implementation
"""

from __future__ import annotations

from context_core.semantic.stores.base import SearchResult, VectorStore
from context_core.semantic.stores.memory import InMemoryVectorStore
from context_core.semantic.stores.qdrant import QdrantVectorStore

__all__ = [
    "InMemoryVectorStore",
    "QdrantVectorStore",
    "SearchResult",
    "VectorStore",
]
