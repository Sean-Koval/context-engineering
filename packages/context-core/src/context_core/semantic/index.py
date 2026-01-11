"""Semantic index for context nodes.

This module provides SemanticIndex, the main class for semantic
indexing and search of context nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from context_core.semantic.embeddings import EmbeddingModel
from context_core.semantic.stores.base import SearchResult, VectorStore
from context_core.semantic.stores.memory import InMemoryVectorStore

if TYPE_CHECKING:
    from context_core.graph.nodes import ContextNode


class SemanticIndex:
    """Semantic indexing for context nodes.

    Provides embedding-based indexing and search for context nodes,
    enabling semantic similarity search, duplicate detection, and
    clustering.

    Features:
    - Index node content as embeddings
    - Semantic similarity search
    - Near-duplicate detection
    - Metadata filtering

    Attributes:
        embedding_model: The embedding model used for vectorization.
        vector_store: The underlying vector storage backend.

    Example:
        >>> from context_core.semantic import SemanticIndex, MockEmbeddingModel
        >>> model = MockEmbeddingModel(dimension=64)
        >>> index = SemanticIndex(model)
        >>> index.index_node(node)
        >>> results = index.search("find similar content", k=5)
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize the semantic index.

        Args:
            embedding_model: Model to generate embeddings from text.
            vector_store: Optional vector store backend.
                If None, uses InMemoryVectorStore.
        """
        self._embedding_model = embedding_model
        self._vector_store: VectorStore = (
            vector_store
            if vector_store is not None
            else InMemoryVectorStore(dimension=embedding_model.dimension)
        )
        self._node_texts: dict[UUID, str] = {}  # Cache of node text content

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Return the embedding model."""
        return self._embedding_model

    @property
    def vector_store(self) -> VectorStore:
        """Return the vector store."""
        return self._vector_store

    def index_node(self, node: ContextNode) -> bool:
        """Index a single node.

        Extracts text content from the node, generates an embedding,
        and stores it in the vector store.

        Args:
            node: The context node to index.

        Returns:
            True if the node was indexed, False if no indexable content.
        """
        text = self._extract_text(node)
        if not text:
            return False

        embedding = self._embedding_model.embed([text])[0]
        metadata = self._build_metadata(node)

        self._vector_store.add(
            ids=[node.id],
            embeddings=np.expand_dims(embedding, 0),
            metadata=[metadata],
        )
        self._node_texts[node.id] = text
        return True

    def index_nodes(self, nodes: list[ContextNode]) -> int:
        """Batch index multiple nodes.

        More efficient than calling index_node repeatedly as it
        batches the embedding computation.

        Args:
            nodes: List of context nodes to index.

        Returns:
            Number of nodes that were successfully indexed.
        """
        texts: list[str] = []
        valid_nodes: list[ContextNode] = []

        for node in nodes:
            text = self._extract_text(node)
            if text:
                texts.append(text)
                valid_nodes.append(node)

        if not texts:
            return 0

        embeddings = self._embedding_model.embed(texts)
        metadata = [self._build_metadata(n) for n in valid_nodes]

        self._vector_store.add(
            ids=[n.id for n in valid_nodes],
            embeddings=embeddings,
            metadata=metadata,
        )

        for node, text in zip(valid_nodes, texts, strict=True):
            self._node_texts[node.id] = text

        return len(valid_nodes)

    def _extract_text(self, node: ContextNode) -> str | None:
        """Extract indexable text from a node.

        Attempts to extract meaningful text content from various
        node content types.

        Args:
            node: The context node to extract text from.

        Returns:
            Extracted text, or None if no indexable content.
        """
        content = node.content

        # Try text content first
        if content.text:
            return content.text

        # Tool calls: combine name and args
        if content.tool_name:
            args_str = str(content.tool_args) if content.tool_args else ""
            return f"{content.tool_name}: {args_str}"

        # Tool output: stringify and truncate
        if content.tool_output is not None:
            return str(content.tool_output)[:1000]

        # Artifact data: stringify and truncate
        if content.artifact_data is not None:
            return str(content.artifact_data)[:1000]

        return None

    def _build_metadata(self, node: ContextNode) -> dict[str, Any]:
        """Build metadata dict for a node.

        Args:
            node: The context node to extract metadata from.

        Returns:
            Metadata dictionary for vector store.
        """
        return {
            "type": node.type.value,
            "compression_level": node.compression_level.value,
            "role": node.content.role.value if node.content.role else None,
        }

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for semantically similar nodes.

        Args:
            query: Search query text.
            k: Maximum number of results to return.
            min_score: Minimum similarity score (0-1) to include.
            filter: Optional metadata filter (e.g., {"type": "message"}).

        Returns:
            List of SearchResult with node IDs and similarity scores.

        Example:
            >>> results = index.search("error handling", k=5)
            >>> for r in results:
            ...     print(f"{r.id}: {r.score:.2f}")
        """
        query_embedding = self._embedding_model.embed([query])[0]
        results = self._vector_store.search(query_embedding, k=k, filter=filter)
        return [r for r in results if r.score >= min_score]

    def search_by_node(
        self,
        node: ContextNode,
        k: int = 10,
        min_score: float = 0.0,
        exclude_self: bool = True,
    ) -> list[SearchResult]:
        """Find nodes similar to a given node.

        Args:
            node: The reference node to find similar nodes for.
            k: Maximum number of results to return.
            min_score: Minimum similarity score to include.
            exclude_self: If True, exclude the query node from results.

        Returns:
            List of SearchResult with similar node IDs.
        """
        text = self._extract_text(node)
        if not text:
            return []

        results = self.search(
            text, k=k + (1 if exclude_self else 0), min_score=min_score
        )

        if exclude_self:
            results = [r for r in results if r.id != node.id][:k]

        return results

    def find_duplicates(
        self,
        threshold: float = 0.95,
    ) -> list[tuple[UUID, UUID, float]]:
        """Find near-duplicate nodes.

        Computes pairwise similarity between all indexed nodes
        and returns pairs that exceed the similarity threshold.

        Args:
            threshold: Minimum similarity score to consider as duplicate (0-1).

        Returns:
            List of (node_id_1, node_id_2, similarity) tuples,
            sorted by similarity descending.

        Example:
            >>> duplicates = index.find_duplicates(threshold=0.9)
            >>> for id1, id2, score in duplicates:
            ...     print(f"Duplicate pair: {score:.2f}")
        """
        node_ids = list(self._node_texts.keys())
        if len(node_ids) < 2:
            return []

        # Get all embeddings
        embeddings = self._vector_store.get(node_ids)
        if len(embeddings) < 2:
            return []

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings_norm = embeddings / norms

        # Compute pairwise similarity matrix
        similarity_matrix = embeddings_norm @ embeddings_norm.T

        # Find pairs above threshold (only upper triangle to avoid duplicates)
        duplicates: list[tuple[UUID, UUID, float]] = []
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                sim = float(similarity_matrix[i, j])
                if sim >= threshold:
                    duplicates.append((node_ids[i], node_ids[j], sim))

        # Sort by similarity descending
        return sorted(duplicates, key=lambda x: x[2], reverse=True)

    def remove_node(self, node_id: UUID) -> bool:
        """Remove a node from the index.

        Args:
            node_id: The ID of the node to remove.

        Returns:
            True if the node was removed, False if it wasn't indexed.
        """
        if node_id not in self._node_texts:
            return False

        self._vector_store.delete([node_id])
        del self._node_texts[node_id]
        return True

    def get_text(self, node_id: UUID) -> str | None:
        """Get the indexed text for a node.

        Args:
            node_id: The ID of the node.

        Returns:
            The indexed text, or None if not found.
        """
        return self._node_texts.get(node_id)

    def is_indexed(self, node_id: UUID) -> bool:
        """Check if a node is indexed.

        Args:
            node_id: The ID of the node to check.

        Returns:
            True if the node is indexed.
        """
        return node_id in self._node_texts

    def count(self) -> int:
        """Return the number of indexed nodes."""
        return len(self._node_texts)

    def clear(self) -> None:
        """Remove all nodes from the index."""
        self._vector_store.clear()
        self._node_texts.clear()

    def __len__(self) -> int:
        """Return the number of indexed nodes."""
        return self.count()

    def __contains__(self, node_id: UUID) -> bool:
        """Check if a node is indexed."""
        return self.is_indexed(node_id)
