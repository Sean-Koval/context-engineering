"""Semantic module - Embedding-based search and similarity.

Components:
- SemanticIndex: Index and search context by meaning
- EmbeddingModel: Protocol for embedding implementations
- SentenceTransformerEmbedding: sentence-transformers implementation
- MockEmbeddingModel: Mock for testing
- VectorStore: Protocol for vector storage backends
- SearchResult: Result model for vector search
- InMemoryVectorStore: Simple in-memory vector store
- QdrantVectorStore: Qdrant-backed vector store

Example:
    >>> from context_core.semantic import SemanticIndex, MockEmbeddingModel
    >>> model = MockEmbeddingModel(dimension=64)
    >>> index = SemanticIndex(model)
    >>> index.index_node(node)
    >>> results = index.search("similar content", k=5)
"""

from __future__ import annotations

from context_core.semantic.embeddings import (
    EmbeddingModel,
    MockEmbeddingModel,
    SentenceTransformerEmbedding,
)
from context_core.semantic.index import SemanticIndex
from context_core.semantic.stores import (
    InMemoryVectorStore,
    QdrantVectorStore,
    SearchResult,
    VectorStore,
)

__all__ = [
    "EmbeddingModel",
    "InMemoryVectorStore",
    "MockEmbeddingModel",
    "QdrantVectorStore",
    "SearchResult",
    "SemanticIndex",
    "SentenceTransformerEmbedding",
    "VectorStore",
]
