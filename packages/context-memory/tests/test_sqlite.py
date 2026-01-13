"""Unit tests for SQLiteStore backend."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from context_memory.backends.sqlite import SQLiteStore
from context_memory.store import MemoryStore
from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageTier,
)


class MockNodeMetadata:
    """Mock node metadata for testing."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNode:
    """Mock ContextNode for testing without importing context-core."""

    def __init__(
        self,
        node_type: str = "MESSAGE",
        content: str = "test content",
        token_count: int = 10,
        importance: float = 0.5,
        tags: set[str] | None = None,
    ) -> None:
        self.id = uuid4()
        self.type = type("NodeType", (), {"value": node_type})()
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance, tags)

    def model_dump(self, mode: str = "python") -> dict:
        """Serialize node to dict."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": {
                "importance": self.metadata.importance,
                "tags": list(self.metadata.tags),
            },
        }


class TestSQLiteStoreInit:
    """Tests for SQLiteStore initialization."""

    @pytest.mark.asyncio
    async def test_creates_database_file(self, tmp_path: Path) -> None:
        """Test that database file is created."""
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        await store.initialize()

        assert db_path.exists()
        await store.close()

    @pytest.mark.asyncio
    async def test_memory_database(self) -> None:
        """Test in-memory database."""
        store = SQLiteStore(":memory:")
        await store.initialize()

        # Should work without errors
        node = MockNode()
        key = await store.store(node, "test-session")
        assert await store.exists(key)

        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, tmp_path: Path) -> None:
        """Test that initialize can be called multiple times."""
        store = SQLiteStore(tmp_path / "test.db")
        await store.initialize()
        await store.initialize()  # Should not raise

        await store.close()

    @pytest.mark.asyncio
    async def test_auto_initialize_on_store(self, tmp_path: Path) -> None:
        """Test that store auto-initializes if needed."""
        store = SQLiteStore(tmp_path / "test.db")
        # Don't call initialize()

        node = MockNode()
        key = await store.store(node, "test-session")

        assert key is not None
        await store.close()

    @pytest.mark.asyncio
    async def test_satisfies_memory_store_protocol(self, tmp_path: Path) -> None:
        """Test that SQLiteStore satisfies MemoryStore protocol."""
        store = SQLiteStore(tmp_path / "test.db")
        assert isinstance(store, MemoryStore)


@pytest.mark.asyncio
class TestSQLiteStoreOperations:
    """Tests for SQLiteStore async operations."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncGenerator[SQLiteStore, None]:
        """Create a SQLiteStore for testing."""
        s = SQLiteStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    async def test_store_and_retrieve(self, store: SQLiteStore) -> None:
        """Test basic store and retrieve."""
        node = MockNode(content="Hello World")
        key = await store.store(node, "test-session")

        assert key.session_id == "test-session"
        assert key.node_id == node.id

        # Retrieve returns dict when context-core not available
        retrieved = await store.retrieve(key)
        assert retrieved is not None
        assert retrieved["content"] == "Hello World"

    async def test_store_with_custom_metadata(self, store: SQLiteStore) -> None:
        """Test storing with custom metadata."""
        node = MockNode()
        custom_meta = StorageMetadata(
            key=StorageKey(session_id="test", node_id=node.id),
            tier=StorageTier.COLD,
            size_bytes=500,
            token_count=50,
            node_type="CUSTOM",
            importance=0.9,
            tags={"custom", "test"},
        )

        key = await store.store(node, "test-session", metadata=custom_meta)
        metadata = await store.get_metadata(key)

        assert metadata is not None
        assert metadata.tier == StorageTier.COLD
        assert metadata.importance == 0.9
        assert "custom" in metadata.tags

    async def test_store_batch(self, store: SQLiteStore) -> None:
        """Test batch store operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(10)]

        keys = await store.store_batch(nodes, "test-session")

        assert len(keys) == 10
        assert all(k.session_id == "test-session" for k in keys)

        # Verify all exist
        for key in keys:
            assert await store.exists(key)

    async def test_store_batch_performance(self, store: SQLiteStore) -> None:
        """Test that batch is faster than individual stores."""
        import time

        nodes = [MockNode(content=f"Node {i}") for i in range(100)]

        # Batch store
        start = time.perf_counter()
        await store.store_batch(nodes, "batch-session")
        batch_time = time.perf_counter() - start

        # Individual stores
        nodes2 = [MockNode(content=f"Node {i}") for i in range(100)]
        start = time.perf_counter()
        for node in nodes2:
            await store.store(node, "individual-session")
        individual_time = time.perf_counter() - start

        # Batch should be faster (or at least not much slower)
        assert batch_time <= individual_time * 2  # Allow some variance

    async def test_retrieve_batch(self, store: SQLiteStore) -> None:
        """Test batch retrieve operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(5)]
        keys = await store.store_batch(nodes, "test-session")

        # Retrieve all
        results = await store.retrieve_batch(keys)

        assert len(results) == 5
        assert all(r is not None for r in results)

    async def test_retrieve_batch_with_missing(self, store: SQLiteStore) -> None:
        """Test batch retrieve with some missing keys."""
        node = MockNode()
        key = await store.store(node, "test-session")

        missing_key = StorageKey(session_id="test", node_id=uuid4())
        results = await store.retrieve_batch([key, missing_key])

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    async def test_exists(self, store: SQLiteStore) -> None:
        """Test exists operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key)

        missing = StorageKey(session_id="test", node_id=uuid4())
        assert not await store.exists(missing)

    async def test_delete(self, store: SQLiteStore) -> None:
        """Test delete operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key)

        deleted = await store.delete(key)
        assert deleted is True
        assert not await store.exists(key)

    async def test_delete_missing(self, store: SQLiteStore) -> None:
        """Test delete on non-existent key."""
        key = StorageKey(session_id="test", node_id=uuid4())
        deleted = await store.delete(key)
        assert deleted is False

    async def test_upsert_behavior(self, store: SQLiteStore) -> None:
        """Test that store updates existing nodes."""
        node = MockNode(content="Original")
        key1 = await store.store(node, "test-session")

        # Store same node with different content
        node.content = "Updated"
        key2 = await store.store(node, "test-session")

        # Should be same key
        assert key1.node_id == key2.node_id

        # Should have updated content
        retrieved = await store.retrieve(key1)
        assert retrieved is not None
        assert retrieved["content"] == "Updated"


@pytest.mark.asyncio
class TestSQLiteStoreMetadata:
    """Tests for metadata operations."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncGenerator[SQLiteStore, None]:
        """Create a SQLiteStore for testing."""
        s = SQLiteStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    async def test_get_metadata(self, store: SQLiteStore) -> None:
        """Test getting metadata."""
        node = MockNode(token_count=42, importance=0.7)
        key = await store.store(node, "test-session")

        metadata = await store.get_metadata(key)

        assert metadata is not None
        assert metadata.token_count == 42
        assert metadata.tier == StorageTier.WARM

    async def test_get_metadata_missing(self, store: SQLiteStore) -> None:
        """Test getting metadata for missing key."""
        key = StorageKey(session_id="test", node_id=uuid4())
        metadata = await store.get_metadata(key)
        assert metadata is None

    async def test_update_metadata(self, store: SQLiteStore) -> None:
        """Test updating metadata."""
        node = MockNode()
        key = await store.store(node, "test-session")

        updated = await store.update_metadata(key, {"importance": 0.95})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.95

    async def test_update_metadata_tags(self, store: SQLiteStore) -> None:
        """Test updating metadata tags."""
        node = MockNode()
        key = await store.store(node, "test-session")

        await store.update_metadata(key, {"tags": ["new", "tags"]})

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert "new" in metadata.tags

    async def test_update_metadata_missing(self, store: SQLiteStore) -> None:
        """Test updating non-existent key."""
        key = StorageKey(session_id="test", node_id=uuid4())
        updated = await store.update_metadata(key, {"importance": 0.5})
        assert updated is False

    async def test_access_tracking(self, store: SQLiteStore) -> None:
        """Test that retrieve updates access tracking."""
        node = MockNode()
        key = await store.store(node, "test-session")

        metadata1 = await store.get_metadata(key)
        assert metadata1 is not None
        initial_count = metadata1.access_count

        # Retrieve updates access count
        await store.retrieve(key)
        await store.retrieve(key)

        metadata2 = await store.get_metadata(key)
        assert metadata2 is not None
        assert metadata2.access_count == initial_count + 2


@pytest.mark.asyncio
class TestSQLiteStoreQueries:
    """Tests for query operations."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncGenerator[SQLiteStore, None]:
        """Create a SQLiteStore for testing."""
        s = SQLiteStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    async def test_list_keys_empty(self, store: SQLiteStore) -> None:
        """Test list_keys on empty session."""
        keys = await store.list_keys("empty-session")
        assert keys == []

    async def test_list_keys(self, store: SQLiteStore) -> None:
        """Test list_keys returns all keys."""
        nodes = [MockNode() for _ in range(5)]
        stored_keys = await store.store_batch(nodes, "test-session")

        listed = await store.list_keys("test-session")

        assert len(listed) == 5
        stored_ids = {k.node_id for k in stored_keys}
        listed_ids = {k.node_id for k in listed}
        assert stored_ids == listed_ids

    async def test_list_keys_limit(self, store: SQLiteStore) -> None:
        """Test list_keys respects limit."""
        nodes = [MockNode() for _ in range(10)]
        await store.store_batch(nodes, "test-session")

        keys = await store.list_keys("test-session", limit=5)
        assert len(keys) == 5

    async def test_list_keys_filter_tier(self, store: SQLiteStore) -> None:
        """Test list_keys filters by tier."""
        node = MockNode()
        key = await store.store(node, "test-session")

        # Update tier
        await store.update_metadata(key, {"tier": "cold"})

        warm_keys = await store.list_keys("test-session", tier=StorageTier.WARM)
        cold_keys = await store.list_keys("test-session", tier=StorageTier.COLD)

        assert len(warm_keys) == 0
        assert len(cold_keys) == 1

    async def test_list_keys_filter_type(self, store: SQLiteStore) -> None:
        """Test list_keys filters by node_type."""
        for _ in range(3):
            node = MockNode(node_type="MESSAGE")
            await store.store(node, "test-session")

        for _ in range(2):
            node = MockNode(node_type="TOOL_CALL")
            await store.store(node, "test-session")

        messages = await store.list_keys("test-session", node_type="MESSAGE")
        assert len(messages) == 3

    async def test_search_by_importance(self, store: SQLiteStore) -> None:
        """Test search_by_metadata filters by importance."""
        for i in range(5):
            node = MockNode(importance=i * 0.2)
            await store.store(node, "test-session")

        results = await store.search_by_metadata("test-session", min_importance=0.5)

        assert len(results) >= 2
        for _key, meta in results:
            assert meta.importance >= 0.5

    async def test_search_by_tags(self, store: SQLiteStore) -> None:
        """Test search_by_metadata filters by tags."""
        node1 = MockNode(tags={"important"})
        key1 = await store.store(node1, "test-session")
        await store.update_metadata(key1, {"tags": ["important", "review"]})

        node2 = MockNode(tags={"other"})
        key2 = await store.store(node2, "test-session")
        await store.update_metadata(key2, {"tags": ["other"]})

        results = await store.search_by_metadata("test-session", tags={"important"})

        assert len(results) >= 1

    async def test_search_by_since(self, store: SQLiteStore) -> None:
        """Test search_by_metadata filters by time."""
        node = MockNode()
        await store.store(node, "test-session")

        # Future time should return nothing
        future = datetime.now(UTC) + timedelta(hours=1)
        results = await store.search_by_metadata("test-session", since=future)
        assert len(results) == 0

        # Past time should return node
        past = datetime.now(UTC) - timedelta(hours=1)
        results = await store.search_by_metadata("test-session", since=past)
        assert len(results) == 1

    async def test_search_ordered_by_importance(self, store: SQLiteStore) -> None:
        """Test search results are ordered by importance descending."""
        for i in range(5):
            node = MockNode(importance=i * 0.2)
            await store.store(node, "test-session")

        results = await store.search_by_metadata("test-session")

        importances = [meta.importance for _key, meta in results]
        assert importances == sorted(importances, reverse=True)


@pytest.mark.asyncio
class TestSQLiteStoreStats:
    """Tests for statistics operations."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncGenerator[SQLiteStore, None]:
        """Create a SQLiteStore for testing."""
        s = SQLiteStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    async def test_stats_empty(self, store: SQLiteStore) -> None:
        """Test stats on empty store."""
        stats = await store.stats("empty-session")

        assert stats.total_items == 0
        assert stats.total_size_bytes == 0
        assert stats.total_tokens == 0

    async def test_stats_with_data(self, store: SQLiteStore) -> None:
        """Test stats with stored data."""
        for _ in range(5):
            node = MockNode(token_count=100)
            await store.store(node, "test-session")

        stats = await store.stats("test-session")

        assert stats.total_items == 5
        assert stats.total_tokens == 500
        assert stats.total_size_bytes > 0

    async def test_stats_tier_breakdown(self, store: SQLiteStore) -> None:
        """Test stats includes tier breakdown."""
        node = MockNode()
        await store.store(node, "test-session")

        stats = await store.stats("test-session")

        assert "warm" in stats.items_by_tier
        assert stats.items_by_tier["warm"] == 1

    async def test_stats_temporal_bounds(self, store: SQLiteStore) -> None:
        """Test stats includes temporal bounds."""
        for _ in range(3):
            node = MockNode()
            await store.store(node, "test-session")

        stats = await store.stats("test-session")

        assert stats.oldest_item is not None
        assert stats.newest_item is not None
        assert stats.oldest_item <= stats.newest_item

    async def test_stats_all_sessions(self, store: SQLiteStore) -> None:
        """Test stats across all sessions."""
        for _ in range(3):
            node = MockNode()
            await store.store(node, "session-1")

        for _ in range(2):
            node = MockNode()
            await store.store(node, "session-2")

        # Stats for specific session
        stats1 = await store.stats("session-1")
        assert stats1.total_items == 3

        stats2 = await store.stats("session-2")
        assert stats2.total_items == 2

        # Stats for all sessions
        all_stats = await store.stats()
        assert all_stats.total_items == 5


@pytest.mark.asyncio
class TestSQLiteStoreClose:
    """Tests for close operation."""

    async def test_close_is_safe(self, tmp_path: Path) -> None:
        """Test that close can be called multiple times."""
        store = SQLiteStore(tmp_path / "test.db")
        await store.initialize()

        await store.close()
        await store.close()  # Should not raise

    async def test_operations_after_close_reinitialize(self, tmp_path: Path) -> None:
        """Test that operations work after close and reinitialize."""
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        await store.initialize()

        node = MockNode()
        key = await store.store(node, "test-session")

        await store.close()

        # Reopen
        store2 = SQLiteStore(db_path)
        await store2.initialize()

        # Data should persist
        assert await store2.exists(key)
        await store2.close()


@pytest.mark.asyncio
class TestSQLiteStoreIntegration:
    """Integration tests for SQLiteStore."""

    @pytest_asyncio.fixture
    async def store(self, tmp_path: Path) -> AsyncGenerator[SQLiteStore, None]:
        """Create a SQLiteStore for testing."""
        s = SQLiteStore(tmp_path / "test.db")
        await s.initialize()
        yield s
        await s.close()

    async def test_full_lifecycle(self, store: SQLiteStore) -> None:
        """Test complete lifecycle."""
        # Store
        node = MockNode(content="Test", token_count=50)
        key = await store.store(node, "lifecycle-session")

        # Verify
        assert await store.exists(key)

        # Get metadata
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.token_count == 50

        # Update metadata
        await store.update_metadata(key, {"importance": 0.9})
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.9

        # List keys
        keys = await store.list_keys("lifecycle-session")
        assert len(keys) == 1

        # Stats
        stats = await store.stats("lifecycle-session")
        assert stats.total_items == 1

        # Delete
        deleted = await store.delete(key)
        assert deleted is True
        assert not await store.exists(key)

        # Stats after delete
        stats = await store.stats("lifecycle-session")
        assert stats.total_items == 0

    async def test_session_isolation(self, store: SQLiteStore) -> None:
        """Test that sessions are isolated."""
        node1 = MockNode()
        node2 = MockNode()

        await store.store(node1, "session-1")
        await store.store(node2, "session-2")

        keys1 = await store.list_keys("session-1")
        keys2 = await store.list_keys("session-2")

        assert len(keys1) == 1
        assert len(keys2) == 1
        assert keys1[0].node_id != keys2[0].node_id

        stats1 = await store.stats("session-1")
        stats2 = await store.stats("session-2")

        assert stats1.total_items == 1
        assert stats2.total_items == 1

    async def test_concurrent_access(self, tmp_path: Path) -> None:
        """Test concurrent access to same database."""
        import asyncio

        db_path = tmp_path / "concurrent.db"
        store = SQLiteStore(db_path)
        await store.initialize()

        async def store_nodes(prefix: str) -> list[StorageKey]:
            keys = []
            for i in range(10):
                node = MockNode(content=f"{prefix}-{i}")
                key = await store.store(node, f"{prefix}-session")
                keys.append(key)
            return keys

        # Run concurrent stores
        results = await asyncio.gather(
            store_nodes("worker-1"),
            store_nodes("worker-2"),
            store_nodes("worker-3"),
        )

        # All should succeed
        assert len(results) == 3
        assert all(len(keys) == 10 for keys in results)

        # Verify data
        stats = await store.stats()
        assert stats.total_items == 30

        await store.close()
