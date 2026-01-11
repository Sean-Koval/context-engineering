"""In-memory vector store implementation.

This module provides InMemoryVectorStore, a simple numpy-based
vector store suitable for development and small-scale use.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from context_core.semantic.stores.base import SearchResult


class InMemoryVectorStore:
    """Simple in-memory vector store using numpy.

    Uses cosine similarity for search. Suitable for development
    and small-scale use (up to ~100k vectors).

    Attributes:
        dimension: The expected dimension of vectors.

    Example:
        >>> store = InMemoryVectorStore(dimension=384)
        >>> store.add([uuid4()], embeddings)
        >>> results = store.search(query, k=5)
    """

    def __init__(self, dimension: int) -> None:
        """Initialize the vector store.

        Args:
            dimension: The dimension of vectors to store.
        """
        self._dimension = dimension
        self._vectors: dict[UUID, NDArray[np.float32]] = {}
        self._metadata: dict[UUID, dict[str, Any]] = {}

    @property
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self._dimension

    def add(
        self,
        ids: list[UUID],
        embeddings: NDArray[np.float32],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the store.

        Args:
            ids: List of unique identifiers for each vector.
            embeddings: Array of shape (n, dimension) containing vectors.
            metadata: Optional list of metadata dicts for each vector.

        Raises:
            ValueError: If ids and embeddings lengths don't match.
            ValueError: If embedding dimension doesn't match store dimension.
        """
        if not ids:
            return

        if len(ids) != len(embeddings):
            msg = (
                f"ids length ({len(ids)}) must match "
                f"embeddings length ({len(embeddings)})"
            )
            raise ValueError(msg)

        if embeddings.shape[1] != self._dimension:
            msg = (
                f"Embedding dimension ({embeddings.shape[1]}) must match "
                f"store dimension ({self._dimension})"
            )
            raise ValueError(msg)

        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            msg = (
                f"metadata length ({len(metadata)}) must match ids length ({len(ids)})"
            )
            raise ValueError(msg)

        for i, id_ in enumerate(ids):
            self._vectors[id_] = embeddings[i].astype(np.float32)
            self._metadata[id_] = metadata[i]

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors using cosine similarity.

        Args:
            query: Query vector of shape (dimension,).
            k: Maximum number of results to return.
            filter: Optional metadata filter (exact match on all keys).

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        if not self._vectors:
            return []

        if query.shape[0] != self._dimension:
            msg = (
                f"Query dimension ({query.shape[0]}) must match "
                f"store dimension ({self._dimension})"
            )
            raise ValueError(msg)

        # Get all vectors and IDs
        ids = list(self._vectors.keys())
        vectors = np.stack([self._vectors[id_] for id_ in ids])

        # Compute cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        similarities = vectors_norm @ query_norm

        # Apply filter if provided
        if filter:
            mask = np.ones(len(ids), dtype=bool)
            for key, value in filter.items():
                for i, id_ in enumerate(ids):
                    if self._metadata[id_].get(key) != value:
                        mask[i] = False
            # Set filtered items to -inf so they sort to the end
            similarities = np.where(mask, similarities, -np.inf)

        # Get top k indices
        if k >= len(ids):
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency with large arrays
            partition_idx = np.argpartition(similarities, -k)[-k:]
            top_indices = partition_idx[np.argsort(similarities[partition_idx])[::-1]]

        # Build results (exclude filtered items with -inf similarity)
        results = []
        for i in top_indices:
            if similarities[i] > -np.inf:
                results.append(
                    SearchResult(
                        id=ids[i],
                        score=float(similarities[i]),
                        metadata=self._metadata[ids[i]],
                    )
                )

        return results[:k]

    def delete(self, ids: list[UUID]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of IDs to delete. Non-existent IDs are ignored.
        """
        for id_ in ids:
            self._vectors.pop(id_, None)
            self._metadata.pop(id_, None)

    def get(self, ids: list[UUID]) -> NDArray[np.float32]:
        """Get vectors by ID.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            Array of shape (n, dimension) containing the vectors.
            Only includes vectors that exist in the store.
        """
        existing = [id_ for id_ in ids if id_ in self._vectors]
        if not existing:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        return np.stack([self._vectors[id_] for id_ in existing])

    def get_metadata(self, id_: UUID) -> dict[str, Any] | None:
        """Get metadata for a specific ID.

        Args:
            id_: The ID to look up.

        Returns:
            The metadata dict, or None if ID doesn't exist.
        """
        return self._metadata.get(id_)

    def count(self) -> int:
        """Return the number of vectors in the store."""
        return len(self._vectors)

    def clear(self) -> None:
        """Remove all vectors from the store."""
        self._vectors.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.count()

    def __contains__(self, id_: UUID) -> bool:
        """Check if an ID exists in the store."""
        return id_ in self._vectors
