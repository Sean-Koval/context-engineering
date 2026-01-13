"""Unit tests for PostgresStore backend.

These tests require a PostgreSQL instance. Set the TEST_POSTGRES_URL environment
variable to run them, e.g.:
    TEST_POSTGRES_URL=postgresql://postgres:postgres@localhost:5432/test_db

Tests will be skipped if no PostgreSQL connection is available.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

# Skip all tests if asyncpg is not available
asyncpg = pytest.importorskip("asyncpg")

from context_memory.backends.postgres import PostgresStore  # noqa: E402
from context_memory.store import MemoryStore  # noqa: E402
from context_memory.types import (  # noqa: E402
    StorageKey,
    StorageMetadata,
    StorageTier,
)

# Get PostgreSQL URL from environment
POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://postgres:postgres@localhost:5432/test_context_memory",
)


class MockNodeMetadata:
    """Mock node metadata for testing."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNodeType:
    """Mock NodeType enum."""

    def __init__(self, value: str = "MESSAGE") -> None:
        self.value = value


class MockNode:
    """Mock ContextNode for testing without importing context-core."""

    def __init__(
        self,
        content: str = "test content",
        token_count: int = 10,
        importance: float = 0.5,
        tags: set[str] | None = None,
    ) -> None:
        self.id = uuid4()
        self.type = MockNodeType("MESSAGE")
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance, tags)

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        """Serialize to dict like Pydantic model."""
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


async def postgres_available() -> bool:
    """Check if PostgreSQL is available for testing."""
    try:
        conn = await asyncpg.connect(POSTGRES_URL)
        await conn.close()
        return True
    except Exception:
        return False


# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL environment variable not set",
)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for module-scoped fixtures."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def store() -> AsyncGenerator[PostgresStore, None]:
    """Create a PostgresStore for testing with cleanup."""
    s = PostgresStore(POSTGRES_URL, pool_min_size=1, pool_max_size=5)
    await s.initialize()

    # Clean up any existing test data
    async with s._db.acquire() as conn:
        await conn.execute("DELETE FROM context_nodes WHERE session_id LIKE 'test-%'")

    yield s

    # Clean up after test
    async with s._db.acquire() as conn:
        await conn.execute("DELETE FROM context_nodes WHERE session_id LIKE 'test-%'")

    await s.close()


@pytest.mark.asyncio
class TestPostgresStoreInit:
    """Tests for PostgresStore initialization."""

    async def test_creates_connection_pool(self) -> None:
        """Test that initialization creates connection pool."""
        store = PostgresStore(POSTGRES_URL)
        await store.initialize()

        assert store._pool is not None
        assert store._initialized is True

        await store.close()

    async def test_initialize_idempotent(self) -> None:
        """Test that initialize can be called multiple times."""
        store = PostgresStore(POSTGRES_URL)
        await store.initialize()
        await store.initialize()  # Should not raise

        assert store._initialized is True
        await store.close()

    async def test_auto_initialize_on_store(self) -> None:
        """Test that store auto-initializes if needed."""
        store = PostgresStore(POSTGRES_URL)
        node = MockNode()

        # Should auto-initialize
        key = await store.store(node, "test-auto-init")
        assert key is not None

        # Clean up
        await store.delete(key)
        await store.close()

    async def test_satisfies_memory_store_protocol(self) -> None:
        """Test that PostgresStore satisfies MemoryStore protocol."""
        store = PostgresStore(POSTGRES_URL)
        assert isinstance(store, MemoryStore)
        await store.close()


@pytest.mark.asyncio
class TestPostgresStoreOperations:
    """Tests for PostgresStore async operations."""

    async def test_store_and_retrieve(self, store: PostgresStore) -> None:
        """Test basic store and retrieve."""
        node = MockNode(content="Hello World")
        key = await store.store(node, "test-session")

        assert key.session_id == "test-session"
        assert key.node_id == node.id
        assert key.version == 1

        retrieved = await store.retrieve(key)
        assert retrieved is not None
        assert retrieved["content"] == "Hello World"

    async def test_store_with_custom_metadata(self, store: PostgresStore) -> None:
        """Test store with custom metadata."""
        node = MockNode(importance=0.9, tags={"important", "test"})
        key = StorageKey(
            session_id="test-session",
            node_id=node.id,
            version=1,
        )
        custom_metadata = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=42,
            node_type="MESSAGE",
            importance=0.9,
            tags={"important", "test"},
        )

        stored_key = await store.store(node, "test-session", custom_metadata)
        metadata = await store.get_metadata(stored_key)

        assert metadata is not None
        assert metadata.tier == StorageTier.HOT
        assert metadata.importance == 0.9
        assert "important" in metadata.tags

    async def test_store_batch(self, store: PostgresStore) -> None:
        """Test batch store operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(5)]
        keys = await store.store_batch(nodes, "test-batch")

        assert len(keys) == 5
        for i, key in enumerate(keys):
            assert key.node_id == nodes[i].id

    async def test_store_batch_performance(self, store: PostgresStore) -> None:
        """Test that batch store is efficient."""
        import time

        nodes = [MockNode(content=f"Node {i}") for i in range(100)]

        start = time.perf_counter()
        keys = await store.store_batch(nodes, "test-perf")
        elapsed = time.perf_counter() - start

        assert len(keys) == 100
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0

    async def test_retrieve_batch(self, store: PostgresStore) -> None:
        """Test batch retrieve operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(3)]
        keys = await store.store_batch(nodes, "test-batch-retrieve")

        retrieved = await store.retrieve_batch(keys)

        assert len(retrieved) == 3
        for i, node in enumerate(retrieved):
            assert node is not None
            assert node["content"] == f"Node {i}"

    async def test_retrieve_batch_with_missing(self, store: PostgresStore) -> None:
        """Test batch retrieve with some missing keys."""
        node = MockNode()
        key = await store.store(node, "test-session")

        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        results = await store.retrieve_batch([key, missing_key])

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    async def test_exists(self, store: PostgresStore) -> None:
        """Test exists operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key) is True

        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        assert await store.exists(missing_key) is False

    async def test_delete(self, store: PostgresStore) -> None:
        """Test delete operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key) is True
        deleted = await store.delete(key)
        assert deleted is True
        assert await store.exists(key) is False

    async def test_delete_missing(self, store: PostgresStore) -> None:
        """Test delete of non-existent key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        deleted = await store.delete(missing_key)
        assert deleted is False

    async def test_upsert_behavior(self, store: PostgresStore) -> None:
        """Test that store with same key updates existing."""
        node = MockNode(content="Original")
        key1 = await store.store(node, "test-session")

        # Store again with same node ID
        node.content = "Updated"
        key2 = await store.store(node, "test-session")

        # Should be same key
        assert key1.node_id == key2.node_id

        # Should have updated content
        retrieved = await store.retrieve(key1)
        assert retrieved is not None
        assert retrieved["content"] == "Updated"


@pytest.mark.asyncio
class TestPostgresStoreMetadata:
    """Tests for metadata operations."""

    async def test_get_metadata(self, store: PostgresStore) -> None:
        """Test getting metadata."""
        node = MockNode(token_count=42, importance=0.7)
        key = await store.store(node, "test-session")

        metadata = await store.get_metadata(key)

        assert metadata is not None
        assert metadata.key == key
        assert metadata.tier == StorageTier.WARM
        assert metadata.token_count == 42
        assert metadata.importance == 0.7
        assert metadata.node_type == "MESSAGE"
        assert metadata.access_count >= 0

    async def test_get_metadata_missing(self, store: PostgresStore) -> None:
        """Test getting metadata for missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        metadata = await store.get_metadata(missing_key)
        assert metadata is None

    async def test_update_metadata(self, store: PostgresStore) -> None:
        """Test updating metadata fields."""
        node = MockNode(importance=0.5)
        key = await store.store(node, "test-session")

        updated = await store.update_metadata(key, {"importance": 0.9})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.9

    async def test_update_metadata_tags(self, store: PostgresStore) -> None:
        """Test updating tags in metadata."""
        node = MockNode()
        key = await store.store(node, "test-session")

        updated = await store.update_metadata(key, {"tags": {"new", "tags"}})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert "new" in metadata.tags
        assert "tags" in metadata.tags

    async def test_update_metadata_missing(self, store: PostgresStore) -> None:
        """Test updating metadata for missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        updated = await store.update_metadata(missing_key, {"importance": 0.9})
        assert updated is False

    async def test_access_tracking(self, store: PostgresStore) -> None:
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
class TestPostgresStoreQueries:
    """Tests for query operations."""

    async def test_list_keys_empty(self, store: PostgresStore) -> None:
        """Test listing keys for empty session."""
        keys = await store.list_keys("test-nonexistent-session")
        assert keys == []

    async def test_list_keys(self, store: PostgresStore) -> None:
        """Test listing keys for session."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-list")

        keys = await store.list_keys("test-list")
        assert len(keys) == 3

    async def test_list_keys_limit(self, store: PostgresStore) -> None:
        """Test list_keys with limit."""
        nodes = [MockNode() for _ in range(10)]
        await store.store_batch(nodes, "test-list-limit")

        keys = await store.list_keys("test-list-limit", limit=5)
        assert len(keys) == 5

    async def test_list_keys_filter_tier(self, store: PostgresStore) -> None:
        """Test list_keys filtered by tier."""
        node1 = MockNode()
        node2 = MockNode()

        key1 = StorageKey(session_id="test-tier", node_id=node1.id, version=1)
        key2 = StorageKey(session_id="test-tier", node_id=node2.id, version=1)

        await store.store(
            node1,
            "test-tier",
            StorageMetadata(
                key=key1,
                tier=StorageTier.HOT,
                size_bytes=100,
                token_count=10,
                node_type="MESSAGE",
            ),
        )
        await store.store(
            node2,
            "test-tier",
            StorageMetadata(
                key=key2,
                tier=StorageTier.COLD,
                size_bytes=100,
                token_count=10,
                node_type="MESSAGE",
            ),
        )

        hot_keys = await store.list_keys("test-tier", tier=StorageTier.HOT)
        assert len(hot_keys) == 1
        assert hot_keys[0].node_id == node1.id

    async def test_list_keys_filter_type(self, store: PostgresStore) -> None:
        """Test list_keys filtered by node type."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-type")

        keys = await store.list_keys("test-type", node_type="MESSAGE")
        assert len(keys) == 3

        keys = await store.list_keys("test-type", node_type="TOOL_CALL")
        assert len(keys) == 0

    async def test_search_by_importance(self, store: PostgresStore) -> None:
        """Test search by minimum importance."""
        nodes = [
            MockNode(importance=0.3),
            MockNode(importance=0.6),
            MockNode(importance=0.9),
        ]
        await store.store_batch(nodes, "test-importance")

        results = await store.search_by_metadata(
            "test-importance", min_importance=0.5
        )
        assert len(results) == 2

    async def test_search_by_tags(self, store: PostgresStore) -> None:
        """Test search by tags."""
        node1 = MockNode(tags={"urgent", "bug"})
        node2 = MockNode(tags={"feature"})

        await store.store(node1, "test-tags")
        await store.store(node2, "test-tags")

        results = await store.search_by_metadata("test-tags", tags={"urgent"})
        assert len(results) == 1
        assert "urgent" in results[0][1].tags

    async def test_search_by_since(self, store: PostgresStore) -> None:
        """Test search by creation time."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-since")

        # Search for nodes created in the future (should return none)
        future = datetime.now(UTC) + timedelta(hours=1)
        results = await store.search_by_metadata("test-since", since=future)
        assert len(results) == 0

        # Search for nodes created recently (should return all)
        past = datetime.now(UTC) - timedelta(hours=1)
        results = await store.search_by_metadata("test-since", since=past)
        assert len(results) == 3

    async def test_search_ordered_by_importance(self, store: PostgresStore) -> None:
        """Test that search results are ordered by importance."""
        nodes = [
            MockNode(importance=0.3),
            MockNode(importance=0.9),
            MockNode(importance=0.6),
        ]
        await store.store_batch(nodes, "test-order")

        results = await store.search_by_metadata("test-order")
        importances = [r[1].importance for r in results]

        assert importances == sorted(importances, reverse=True)


@pytest.mark.asyncio
class TestPostgresStoreStats:
    """Tests for statistics operations."""

    async def test_stats_empty(self, store: PostgresStore) -> None:
        """Test stats for empty session."""
        stats = await store.stats("test-nonexistent")
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0

    async def test_stats_with_data(self, store: PostgresStore) -> None:
        """Test stats with stored data."""
        nodes = [MockNode(token_count=100) for _ in range(5)]
        await store.store_batch(nodes, "test-stats")

        stats = await store.stats("test-stats")
        assert stats.total_items == 5
        assert stats.total_tokens == 500
        assert stats.total_size_bytes > 0

    async def test_stats_tier_breakdown(self, store: PostgresStore) -> None:
        """Test stats tier breakdown."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-tier-stats")

        stats = await store.stats("test-tier-stats")
        assert StorageTier.WARM.value in stats.items_by_tier
        assert stats.items_by_tier[StorageTier.WARM.value] == 3

    async def test_stats_temporal_bounds(self, store: PostgresStore) -> None:
        """Test stats temporal bounds."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-temporal")

        stats = await store.stats("test-temporal")
        assert stats.oldest_item is not None
        assert stats.newest_item is not None
        assert stats.oldest_item <= stats.newest_item

    async def test_stats_all_sessions(self, store: PostgresStore) -> None:
        """Test stats across all sessions."""
        await store.store(MockNode(), "test-stats-a")
        await store.store(MockNode(), "test-stats-b")

        stats = await store.stats()  # No session_id
        assert stats.total_items >= 2


@pytest.mark.asyncio
class TestPostgresStoreClose:
    """Tests for close and lifecycle operations."""

    async def test_close_is_safe(self) -> None:
        """Test that close can be called safely."""
        store = PostgresStore(POSTGRES_URL)
        await store.initialize()
        await store.close()
        await store.close()  # Should not raise

    async def test_operations_after_close_reinitialize(self) -> None:
        """Test that store can be reused after close."""
        store = PostgresStore(POSTGRES_URL)
        await store.initialize()
        await store.close()

        # Should auto-reinitialize
        node = MockNode()
        key = await store.store(node, "test-reopen")
        assert key is not None

        await store.delete(key)
        await store.close()


@pytest.mark.asyncio
class TestPostgresStoreIntegration:
    """Integration tests for full workflows."""

    async def test_full_lifecycle(self, store: PostgresStore) -> None:
        """Test complete storage lifecycle."""
        # Store
        node = MockNode(content="Integration test", importance=0.8)
        key = await store.store(node, "test-integration")

        # Retrieve
        retrieved = await store.retrieve(key)
        assert retrieved is not None
        assert retrieved["content"] == "Integration test"

        # Get metadata
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.8

        # Update metadata
        await store.update_metadata(key, {"importance": 1.0})
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 1.0

        # Search
        results = await store.search_by_metadata(
            "test-integration", min_importance=0.9
        )
        assert len(results) == 1

        # Stats
        stats = await store.stats("test-integration")
        assert stats.total_items == 1

        # Delete
        deleted = await store.delete(key)
        assert deleted is True
        assert await store.exists(key) is False

    async def test_session_isolation(self, store: PostgresStore) -> None:
        """Test that sessions are isolated."""
        node1 = MockNode(content="Session 1")
        node2 = MockNode(content="Session 2")

        await store.store(node1, "test-isolation-1")
        await store.store(node2, "test-isolation-2")

        keys1 = await store.list_keys("test-isolation-1")
        keys2 = await store.list_keys("test-isolation-2")

        assert len(keys1) == 1
        assert len(keys2) == 1
        assert keys1[0].node_id != keys2[0].node_id

    async def test_concurrent_access(self, store: PostgresStore) -> None:
        """Test concurrent store operations."""
        import asyncio

        async def store_node(i: int) -> StorageKey:
            node = MockNode(content=f"Concurrent {i}")
            return await store.store(node, "test-concurrent")

        keys = await asyncio.gather(*[store_node(i) for i in range(10)])
        assert len(keys) == 10
        assert len({k.node_id for k in keys}) == 10  # All unique
