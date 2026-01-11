"""Base classes and protocols for vector stores.

This module provides:
- SearchResult: Pydantic model for search results
- VectorStore: Protocol for vector storage backends
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Result from vector search.

    Attributes:
        id: The unique identifier of the matched item.
        score: Similarity score (higher is more similar, typically 0-1).
        metadata: Additional metadata associated with the item.

    Example:
        >>> result = SearchResult(id=uuid4(), score=0.95, metadata={"type": "message"})
        >>> result.score
        0.95
    """

    id: UUID = Field(description="Unique identifier of the matched item")
    score: float = Field(description="Similarity score (higher is better)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    model_config = {"frozen": True}


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    Vector stores provide persistent or in-memory storage for
    embedding vectors with support for similarity search.

    All implementations must provide:
    - add: Add vectors with IDs and optional metadata
    - search: Find similar vectors
    - delete: Remove vectors by ID
    - get: Retrieve vectors by ID

    Example:
        >>> store = InMemoryVectorStore(dimension=384)
        >>> store.add([id1], embeddings, [{"type": "message"}])
        >>> results = store.search(query_embedding, k=5)
    """

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
        """
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query: Query vector of shape (dimension,).
            k: Maximum number of results to return.
            filter: Optional metadata filter (e.g., {"type": "message"}).

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        ...

    def delete(self, ids: list[UUID]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of IDs to delete.
        """
        ...

    def get(self, ids: list[UUID]) -> NDArray[np.float32]:
        """Get vectors by ID.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            Array of shape (n, dimension) containing the vectors.
            Only includes vectors that exist in the store.
        """
        ...

    def count(self) -> int:
        """Return the number of vectors in the store."""
        ...

    def clear(self) -> None:
        """Remove all vectors from the store."""
        ...
