"""Qdrant vector store implementation.

This module provides QdrantVectorStore, a persistent vector store
backed by Qdrant for production use.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from context_core.semantic.stores.base import SearchResult

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class QdrantVectorStore:
    """Vector store backed by Qdrant.

    Provides persistent vector storage with efficient similarity search.
    Supports both in-memory and persistent modes, as well as remote
    Qdrant server connections.

    Attributes:
        collection_name: Name of the Qdrant collection.
        dimension: Vector dimension (set on first add if not specified).

    Example:
        >>> store = QdrantVectorStore(collection_name="context_nodes", dimension=384)
        >>> store.add([uuid4()], embeddings, [{"type": "message"}])
        >>> results = store.search(query, k=5)
    """

    def __init__(
        self,
        collection_name: str = "context_nodes",
        dimension: int | None = None,
        path: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        """Initialize the Qdrant vector store.

        Args:
            collection_name: Name of the collection to use.
            dimension: Vector dimension. Required for creating collection.
            path: Path for local persistent storage.
                If None and no url/client, uses in-memory storage.
            url: URL of remote Qdrant server.
            api_key: API key for remote Qdrant server.
            client: Optional pre-configured QdrantClient.
                If provided, path/url/api_key are ignored.

        Raises:
            ImportError: If qdrant-client is not installed.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            msg = (
                "qdrant-client is required for QdrantVectorStore. "
                "Install with: pip install qdrant-client"
            )
            raise ImportError(msg) from e

        self._collection_name = collection_name
        self._dimension = dimension
        self._Distance = Distance
        self._VectorParams = VectorParams

        if client is not None:
            self._client = client
        elif url:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path:
            self._client = QdrantClient(path=path)
        else:
            # In-memory mode
            self._client = QdrantClient(":memory:")

        # Create collection if dimension is known
        if dimension is not None:
            self._ensure_collection(dimension)

    @property
    def collection_name(self) -> str:
        """Return the collection name."""
        return self._collection_name

    @property
    def dimension(self) -> int | None:
        """Return the vector dimension."""
        return self._dimension

    def _ensure_collection(self, dimension: int) -> None:
        """Ensure collection exists with correct dimension."""
        from qdrant_client.models import Distance, VectorParams

        collections = self._client.get_collections().collections
        exists = any(c.name == self._collection_name for c in collections)

        if not exists:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            )
            self._dimension = dimension

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
        from qdrant_client.models import PointStruct

        if not ids:
            return

        if len(ids) != len(embeddings):
            msg = (
                f"ids length ({len(ids)}) must match "
                f"embeddings length ({len(embeddings)})"
            )
            raise ValueError(msg)

        # Ensure collection exists
        dimension = embeddings.shape[1]
        if self._dimension is None:
            self._ensure_collection(dimension)

        metadata = metadata or [{} for _ in ids]
        if len(metadata) != len(ids):
            msg = (
                f"metadata length ({len(metadata)}) must match ids length ({len(ids)})"
            )
            raise ValueError(msg)

        # Convert to Qdrant points
        points = [
            PointStruct(
                id=str(id_),
                vector=embeddings[i].tolist(),
                payload=metadata[i],
            )
            for i, id_ in enumerate(ids)
        ]

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

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
            filter: Optional metadata filter.

        Returns:
            List of SearchResult ordered by descending similarity.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Build Qdrant filter
        qdrant_filter = None
        if filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query.tolist(),
            limit=k,
            query_filter=qdrant_filter,
        )

        # Convert to SearchResult objects
        return [
            SearchResult(
                id=UUID(point.id) if isinstance(point.id, str) else UUID(int=point.id),
                score=point.score,
                metadata=point.payload or {},
            )
            for point in results
        ]

    def delete(self, ids: list[UUID]) -> None:
        """Delete vectors by ID.

        Args:
            ids: List of IDs to delete.
        """
        from qdrant_client.models import PointIdsList

        if not ids:
            return

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=[str(id_) for id_ in ids]),
        )

    def get(self, ids: list[UUID]) -> NDArray[np.float32]:
        """Get vectors by ID.

        Args:
            ids: List of IDs to retrieve.

        Returns:
            Array of shape (n, dimension) containing the vectors.
        """
        if not ids:
            dim = self._dimension or 0
            return (
                np.array([], dtype=np.float32).reshape(0, dim)
                if dim
                else np.array([], dtype=np.float32)
            )

        results = self._client.retrieve(
            collection_name=self._collection_name,
            ids=[str(id_) for id_ in ids],
            with_vectors=True,
        )

        if not results:
            dim = self._dimension or 0
            return (
                np.array([], dtype=np.float32).reshape(0, dim)
                if dim
                else np.array([], dtype=np.float32)
            )

        vectors = [point.vector for point in results if point.vector]
        if not vectors:
            dim = self._dimension or 0
            return (
                np.array([], dtype=np.float32).reshape(0, dim)
                if dim
                else np.array([], dtype=np.float32)
            )

        return np.array(vectors, dtype=np.float32)

    def count(self) -> int:
        """Return the number of vectors in the store."""
        try:
            info = self._client.get_collection(self._collection_name)
            return info.points_count
        except Exception:
            return 0

    def clear(self) -> None:
        """Remove all vectors from the store."""
        # Delete and recreate collection (suppress errors if collection doesn't exist)
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._collection_name)

        if self._dimension:
            self._ensure_collection(self._dimension)

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.count()
