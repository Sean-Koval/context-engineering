"""Tests for vector stores."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from context_core.semantic.stores.base import SearchResult, VectorStore
from context_core.semantic.stores.memory import InMemoryVectorStore


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_result(self) -> None:
        """Test creating a search result."""
        id_ = uuid4()
        result = SearchResult(id=id_, score=0.95, metadata={"type": "message"})

        assert result.id == id_
        assert result.score == 0.95
        assert result.metadata == {"type": "message"}

    def test_default_metadata(self) -> None:
        """Test default empty metadata."""
        result = SearchResult(id=uuid4(), score=0.5)
        assert result.metadata == {}

    def test_frozen(self) -> None:
        """Test that SearchResult is immutable."""
        result = SearchResult(id=uuid4(), score=0.5)
        with pytest.raises(ValidationError):
            result.score = 0.9  # type: ignore


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore."""

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Create a test store."""
        return InMemoryVectorStore(dimension=64)

    def test_dimension(self, store: InMemoryVectorStore) -> None:
        """Test dimension property."""
        assert store.dimension == 64

    def test_add_single_vector(self, store: InMemoryVectorStore) -> None:
        """Test adding a single vector."""
        id_ = uuid4()
        embedding = np.random.randn(1, 64).astype(np.float32)

        store.add([id_], embedding)
        assert store.count() == 1
        assert id_ in store

    def test_add_multiple_vectors(self, store: InMemoryVectorStore) -> None:
        """Test adding multiple vectors."""
        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)

        store.add(ids, embeddings)
        assert store.count() == 5
        for id_ in ids:
            assert id_ in store

    def test_add_with_metadata(self, store: InMemoryVectorStore) -> None:
        """Test adding vectors with metadata."""
        id_ = uuid4()
        embedding = np.random.randn(1, 64).astype(np.float32)
        metadata = [{"type": "message", "role": "user"}]

        store.add([id_], embedding, metadata)
        assert store.get_metadata(id_) == {"type": "message", "role": "user"}

    def test_add_empty(self, store: InMemoryVectorStore) -> None:
        """Test adding empty list."""
        store.add([], np.array([]).reshape(0, 64).astype(np.float32))
        assert store.count() == 0

    def test_add_mismatched_lengths(self, store: InMemoryVectorStore) -> None:
        """Test error on mismatched lengths."""
        ids = [uuid4(), uuid4()]
        embeddings = np.random.randn(3, 64).astype(np.float32)

        with pytest.raises(ValueError, match="must match"):
            store.add(ids, embeddings)

    def test_add_wrong_dimension(self, store: InMemoryVectorStore) -> None:
        """Test error on wrong dimension."""
        ids = [uuid4()]
        embeddings = np.random.randn(1, 128).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            store.add(ids, embeddings)

    def test_search_basic(self, store: InMemoryVectorStore) -> None:
        """Test basic search."""
        ids = [uuid4() for _ in range(10)]
        embeddings = np.random.randn(10, 64).astype(np.float32)
        store.add(ids, embeddings)

        query = embeddings[0]  # Search for first vector
        results = store.search(query, k=3)

        assert len(results) == 3
        assert results[0].id == ids[0]  # Should find itself
        assert results[0].score > 0.99  # Very high similarity

    def test_search_returns_scores(self, store: InMemoryVectorStore) -> None:
        """Test that search returns valid scores."""
        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)
        store.add(ids, embeddings)

        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=5)

        for r in results:
            assert -1 <= r.score <= 1  # Cosine similarity range

    def test_search_ordered_by_score(self, store: InMemoryVectorStore) -> None:
        """Test that results are ordered by descending score."""
        ids = [uuid4() for _ in range(10)]
        embeddings = np.random.randn(10, 64).astype(np.float32)
        store.add(ids, embeddings)

        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=10)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_filter(self, store: InMemoryVectorStore) -> None:
        """Test search with metadata filter."""
        ids = [uuid4() for _ in range(10)]
        embeddings = np.random.randn(10, 64).astype(np.float32)
        metadata = [{"type": "message" if i % 2 == 0 else "tool"} for i in range(10)]
        store.add(ids, embeddings, metadata)

        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=10, filter={"type": "message"})

        # Should only return message type
        assert len(results) == 5
        for r in results:
            assert r.metadata["type"] == "message"

    def test_search_empty_store(self, store: InMemoryVectorStore) -> None:
        """Test search on empty store."""
        query = np.random.randn(64).astype(np.float32)
        results = store.search(query)
        assert results == []

    def test_search_k_larger_than_store(self, store: InMemoryVectorStore) -> None:
        """Test search with k larger than store size."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, k=10)

        assert len(results) == 3

    def test_delete_single(self, store: InMemoryVectorStore) -> None:
        """Test deleting a single vector."""
        id_ = uuid4()
        embedding = np.random.randn(1, 64).astype(np.float32)
        store.add([id_], embedding)

        store.delete([id_])
        assert store.count() == 0
        assert id_ not in store

    def test_delete_multiple(self, store: InMemoryVectorStore) -> None:
        """Test deleting multiple vectors."""
        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)
        store.add(ids, embeddings)

        store.delete(ids[:3])
        assert store.count() == 2
        assert ids[3] in store
        assert ids[4] in store

    def test_delete_nonexistent(self, store: InMemoryVectorStore) -> None:
        """Test deleting non-existent ID (should not error)."""
        store.delete([uuid4()])
        assert store.count() == 0

    def test_get_vectors(self, store: InMemoryVectorStore) -> None:
        """Test getting vectors by ID."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        retrieved = store.get(ids[:2])
        assert retrieved.shape == (2, 64)
        np.testing.assert_array_equal(retrieved[0], embeddings[0])

    def test_get_empty(self, store: InMemoryVectorStore) -> None:
        """Test getting with empty list."""
        result = store.get([])
        assert result.shape == (0, 64)

    def test_get_nonexistent(self, store: InMemoryVectorStore) -> None:
        """Test getting non-existent IDs."""
        result = store.get([uuid4()])
        assert result.shape == (0, 64)

    def test_clear(self, store: InMemoryVectorStore) -> None:
        """Test clearing the store."""
        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)
        store.add(ids, embeddings)

        store.clear()
        assert store.count() == 0
        assert len(store) == 0

    def test_len(self, store: InMemoryVectorStore) -> None:
        """Test __len__ method."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        assert len(store) == 3

    def test_contains(self, store: InMemoryVectorStore) -> None:
        """Test __contains__ method."""
        id_ = uuid4()
        embedding = np.random.randn(1, 64).astype(np.float32)
        store.add([id_], embedding)

        assert id_ in store
        assert uuid4() not in store

    def test_implements_protocol(self, store: InMemoryVectorStore) -> None:
        """Test that InMemoryVectorStore implements VectorStore protocol."""
        assert isinstance(store, VectorStore)


# Skip Qdrant tests if not installed
try:
    import qdrant_client  # noqa: F401

    from context_core.semantic.stores.qdrant import QdrantVectorStore

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
    QdrantVectorStore = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
class TestQdrantVectorStore:
    """Tests for QdrantVectorStore."""

    @pytest.fixture
    def store(self) -> QdrantVectorStore:
        """Create a test store (in-memory)."""
        return QdrantVectorStore(
            collection_name=f"test_{uuid4().hex[:8]}",
            dimension=64,
        )

    def test_collection_name(self, store: QdrantVectorStore) -> None:
        """Test collection name property."""
        assert store.collection_name.startswith("test_")

    def test_dimension(self, store: QdrantVectorStore) -> None:
        """Test dimension property."""
        assert store.dimension == 64

    def test_add_and_search(self, store: QdrantVectorStore) -> None:
        """Test basic add and search."""
        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)
        store.add(ids, embeddings)

        query = embeddings[0]
        results = store.search(query, k=3)

        assert len(results) >= 1
        assert results[0].id == ids[0]

    def test_add_with_metadata(self, store: QdrantVectorStore) -> None:
        """Test adding with metadata."""
        ids = [uuid4()]
        embeddings = np.random.randn(1, 64).astype(np.float32)
        metadata = [{"type": "message"}]

        store.add(ids, embeddings, metadata)
        results = store.search(embeddings[0], k=1)

        assert results[0].metadata.get("type") == "message"

    def test_search_with_filter(self, store: QdrantVectorStore) -> None:
        """Test search with filter."""
        ids = [uuid4() for _ in range(4)]
        embeddings = np.random.randn(4, 64).astype(np.float32)
        metadata = [
            {"type": "message"},
            {"type": "tool"},
            {"type": "message"},
            {"type": "tool"},
        ]
        store.add(ids, embeddings, metadata)

        results = store.search(embeddings[0], k=10, filter={"type": "message"})
        for r in results:
            assert r.metadata.get("type") == "message"

    def test_delete(self, store: QdrantVectorStore) -> None:
        """Test deletion."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        store.delete([ids[0]])
        assert store.count() == 2

    def test_count(self, store: QdrantVectorStore) -> None:
        """Test count method."""
        assert store.count() == 0

        ids = [uuid4() for _ in range(5)]
        embeddings = np.random.randn(5, 64).astype(np.float32)
        store.add(ids, embeddings)

        assert store.count() == 5

    def test_clear(self, store: QdrantVectorStore) -> None:
        """Test clearing the store."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        store.clear()
        assert store.count() == 0

    def test_get_vectors(self, store: QdrantVectorStore) -> None:
        """Test getting vectors by ID."""
        ids = [uuid4() for _ in range(3)]
        embeddings = np.random.randn(3, 64).astype(np.float32)
        store.add(ids, embeddings)

        retrieved = store.get(ids[:2])
        assert retrieved.shape == (2, 64)

    def test_implements_protocol(self, store: QdrantVectorStore) -> None:
        """Test protocol compliance."""
        assert isinstance(store, VectorStore)
