"""Unit tests for RedisStore backend.

These tests require a Redis instance. Set the TEST_REDIS_URL environment
variable to run them, e.g.:
    TEST_REDIS_URL=redis://localhost:6379

Tests will be skipped if no Redis connection is available.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio

# Skip all tests if redis is not available
redis_lib = pytest.importorskip("redis")

from context_memory.backends.redis import RedisStore  # noqa: E402
from context_memory.store import MemoryStore  # noqa: E402
from context_memory.types import (  # noqa: E402
    StorageKey,
    StorageMetadata,
    StorageTier,
)

# Get Redis URL from environment
REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379")


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


# Skip all tests if Redis is not available
pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_REDIS_URL"),
    reason="TEST_REDIS_URL environment variable not set",
)


@pytest_asyncio.fixture
async def store() -> AsyncGenerator[RedisStore, None]:
    """Create a RedisStore for testing with cleanup."""
    # Use a test-specific key prefix to avoid conflicts
    s = RedisStore(
        REDIS_URL,
        key_prefix="test_ctx",
        ttl_seconds=300,  # 5 minutes for tests
    )
    await s.initialize()

    yield s

    # Clean up test data
    pattern = "test_ctx:test-*"
    async for key in s._redis.scan_iter(match=pattern, count=100):
        await s._redis.delete(key)

    await s.close()


@pytest.mark.asyncio
class TestRedisStoreInit:
    """Tests for RedisStore initialization."""

    async def test_creates_connection(self) -> None:
        """Test that initialization creates connection."""
        store = RedisStore(REDIS_URL, key_prefix="test_init")
        await store.initialize()

        assert store._client is not None
        assert store._initialized is True

        await store.close()

    async def test_initialize_idempotent(self) -> None:
        """Test that initialize can be called multiple times."""
        store = RedisStore(REDIS_URL, key_prefix="test_idem")
        await store.initialize()
        await store.initialize()  # Should not raise

        assert store._initialized is True
        await store.close()

    async def test_auto_initialize_on_store(self) -> None:
        """Test that store auto-initializes if needed."""
        store = RedisStore(REDIS_URL, key_prefix="test_auto")
        node = MockNode()

        # Should auto-initialize
        key = await store.store(node, "test-auto-init")
        assert key is not None

        # Clean up
        await store.delete(key)
        await store.close()

    async def test_satisfies_memory_store_protocol(self) -> None:
        """Test that RedisStore satisfies MemoryStore protocol."""
        store = RedisStore(REDIS_URL)
        assert isinstance(store, MemoryStore)
        await store.close()


@pytest.mark.asyncio
class TestRedisStoreOperations:
    """Tests for RedisStore async operations."""

    async def test_store_and_retrieve(self, store: RedisStore) -> None:
        """Test basic store and retrieve."""
        node = MockNode(content="Hello World")
        key = await store.store(node, "test-session")

        assert key.session_id == "test-session"
        assert key.node_id == node.id
        assert key.version == 1

        retrieved = await store.retrieve(key)
        assert retrieved is not None
        assert retrieved["content"] == "Hello World"

    async def test_store_uses_hot_tier(self, store: RedisStore) -> None:
        """Test that store defaults to HOT tier."""
        node = MockNode()
        key = await store.store(node, "test-session")

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.tier == StorageTier.HOT

    async def test_store_with_custom_metadata(self, store: RedisStore) -> None:
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
        assert metadata.importance == 0.9
        assert "important" in metadata.tags

    async def test_store_batch(self, store: RedisStore) -> None:
        """Test batch store operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(5)]
        keys = await store.store_batch(nodes, "test-batch")

        assert len(keys) == 5
        for i, key in enumerate(keys):
            assert key.node_id == nodes[i].id

    async def test_store_batch_performance(self, store: RedisStore) -> None:
        """Test that batch store is efficient."""
        import time

        nodes = [MockNode(content=f"Node {i}") for i in range(100)]

        start = time.perf_counter()
        keys = await store.store_batch(nodes, "test-perf")
        elapsed = time.perf_counter() - start

        assert len(keys) == 100
        # Redis should be very fast (< 1 second for 100 items)
        assert elapsed < 1.0

    async def test_retrieve_batch(self, store: RedisStore) -> None:
        """Test batch retrieve operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(3)]
        keys = await store.store_batch(nodes, "test-batch-retrieve")

        retrieved = await store.retrieve_batch(keys)

        assert len(retrieved) == 3
        for i, node in enumerate(retrieved):
            assert node is not None
            assert node["content"] == f"Node {i}"

    async def test_retrieve_batch_with_missing(self, store: RedisStore) -> None:
        """Test batch retrieve with some missing keys."""
        node = MockNode()
        key = await store.store(node, "test-session")

        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        results = await store.retrieve_batch([key, missing_key])

        assert len(results) == 2
        assert results[0] is not None
        assert results[1] is None

    async def test_exists(self, store: RedisStore) -> None:
        """Test exists operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key) is True

        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        assert await store.exists(missing_key) is False

    async def test_delete(self, store: RedisStore) -> None:
        """Test delete operation."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key) is True
        deleted = await store.delete(key)
        assert deleted is True
        assert await store.exists(key) is False

    async def test_delete_missing(self, store: RedisStore) -> None:
        """Test delete of non-existent key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        deleted = await store.delete(missing_key)
        assert deleted is False


@pytest.mark.asyncio
class TestRedisStoreMetadata:
    """Tests for metadata operations."""

    async def test_get_metadata(self, store: RedisStore) -> None:
        """Test getting metadata."""
        node = MockNode(token_count=42, importance=0.7)
        key = await store.store(node, "test-session")

        metadata = await store.get_metadata(key)

        assert metadata is not None
        assert metadata.key == key
        assert metadata.tier == StorageTier.HOT
        assert metadata.token_count == 42
        assert metadata.importance == 0.7
        assert metadata.node_type == "MESSAGE"
        assert metadata.access_count >= 0

    async def test_get_metadata_missing(self, store: RedisStore) -> None:
        """Test getting metadata for missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        metadata = await store.get_metadata(missing_key)
        assert metadata is None

    async def test_update_metadata(self, store: RedisStore) -> None:
        """Test updating metadata fields."""
        node = MockNode(importance=0.5)
        key = await store.store(node, "test-session")

        updated = await store.update_metadata(key, {"importance": 0.9})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.9

    async def test_update_metadata_tags(self, store: RedisStore) -> None:
        """Test updating tags in metadata."""
        node = MockNode()
        key = await store.store(node, "test-session")

        updated = await store.update_metadata(key, {"tags": {"new", "tags"}})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert "new" in metadata.tags
        assert "tags" in metadata.tags

    async def test_update_metadata_missing(self, store: RedisStore) -> None:
        """Test updating metadata for missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        updated = await store.update_metadata(missing_key, {"importance": 0.9})
        assert updated is False

    async def test_access_tracking(self, store: RedisStore) -> None:
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
class TestRedisStoreQueries:
    """Tests for query operations."""

    async def test_list_keys_empty(self, store: RedisStore) -> None:
        """Test listing keys for empty session."""
        keys = await store.list_keys("test-nonexistent-session")
        assert keys == []

    async def test_list_keys(self, store: RedisStore) -> None:
        """Test listing keys for session."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-list")

        keys = await store.list_keys("test-list")
        assert len(keys) == 3

    async def test_list_keys_limit(self, store: RedisStore) -> None:
        """Test list_keys with limit."""
        nodes = [MockNode() for _ in range(10)]
        await store.store_batch(nodes, "test-list-limit")

        keys = await store.list_keys("test-list-limit", limit=5)
        assert len(keys) == 5

    async def test_list_keys_filter_tier(self, store: RedisStore) -> None:
        """Test list_keys filtered by tier."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-tier-filter")

        # All should be HOT tier by default
        hot_keys = await store.list_keys("test-tier-filter", tier=StorageTier.HOT)
        assert len(hot_keys) == 3

        cold_keys = await store.list_keys("test-tier-filter", tier=StorageTier.COLD)
        assert len(cold_keys) == 0

    async def test_list_keys_filter_type(self, store: RedisStore) -> None:
        """Test list_keys filtered by node type."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-type-filter")

        keys = await store.list_keys("test-type-filter", node_type="MESSAGE")
        assert len(keys) == 3

        keys = await store.list_keys("test-type-filter", node_type="TOOL_CALL")
        assert len(keys) == 0

    async def test_search_by_importance(self, store: RedisStore) -> None:
        """Test search by minimum importance."""
        nodes = [
            MockNode(importance=0.3),
            MockNode(importance=0.6),
            MockNode(importance=0.9),
        ]
        await store.store_batch(nodes, "test-importance")

        results = await store.search_by_metadata("test-importance", min_importance=0.5)
        assert len(results) == 2

    async def test_search_by_tags(self, store: RedisStore) -> None:
        """Test search by tags."""
        node1 = MockNode(tags={"urgent", "bug"})
        node2 = MockNode(tags={"feature"})

        await store.store(node1, "test-tags")
        await store.store(node2, "test-tags")

        results = await store.search_by_metadata("test-tags", tags={"urgent"})
        assert len(results) == 1
        assert "urgent" in results[0][1].tags

    async def test_search_by_since(self, store: RedisStore) -> None:
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

    async def test_search_ordered_by_importance(self, store: RedisStore) -> None:
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
class TestRedisStoreStats:
    """Tests for statistics operations."""

    async def test_stats_empty(self, store: RedisStore) -> None:
        """Test stats for empty session."""
        stats = await store.stats("test-nonexistent")
        assert stats.total_items == 0
        assert stats.total_size_bytes == 0

    async def test_stats_with_data(self, store: RedisStore) -> None:
        """Test stats with stored data."""
        nodes = [MockNode(token_count=100) for _ in range(5)]
        await store.store_batch(nodes, "test-stats")

        stats = await store.stats("test-stats")
        assert stats.total_items == 5
        assert stats.total_tokens == 500
        assert stats.total_size_bytes > 0

    async def test_stats_tier_breakdown(self, store: RedisStore) -> None:
        """Test stats tier breakdown."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-tier-stats")

        stats = await store.stats("test-tier-stats")
        assert StorageTier.HOT.value in stats.items_by_tier
        assert stats.items_by_tier[StorageTier.HOT.value] == 3

    async def test_stats_temporal_bounds(self, store: RedisStore) -> None:
        """Test stats temporal bounds."""
        nodes = [MockNode() for _ in range(3)]
        await store.store_batch(nodes, "test-temporal")

        stats = await store.stats("test-temporal")
        assert stats.oldest_item is not None
        assert stats.newest_item is not None
        assert stats.oldest_item <= stats.newest_item


@pytest.mark.asyncio
class TestRedisStoreTTL:
    """Tests for TTL-specific operations."""

    async def test_ttl_is_set(self, store: RedisStore) -> None:
        """Test that TTL is set on store."""
        node = MockNode()
        key = await store.store(node, "test-session")

        ttl = await store.get_ttl(key)
        assert ttl is not None
        assert ttl > 0
        assert ttl <= 300  # 5 minutes (test fixture TTL)

    async def test_retrieve_resets_ttl(self, store: RedisStore) -> None:
        """Test that retrieve resets TTL."""
        node = MockNode()
        key = await store.store(node, "test-session")

        # Wait a bit
        import asyncio

        await asyncio.sleep(0.1)

        ttl_before = await store.get_ttl(key)

        # Retrieve resets TTL
        await store.retrieve(key)

        ttl_after = await store.get_ttl(key)
        assert ttl_after is not None
        assert ttl_after >= ttl_before  # Should be reset to max

    async def test_touch_resets_ttl(self, store: RedisStore) -> None:
        """Test that touch resets TTL without retrieving."""
        node = MockNode()
        key = await store.store(node, "test-session")

        touched = await store.touch(key)
        assert touched is True

        # Touch also increments access count
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.access_count >= 1

    async def test_touch_missing_key(self, store: RedisStore) -> None:
        """Test touch on missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        touched = await store.touch(missing_key)
        assert touched is False

    async def test_set_ttl(self, store: RedisStore) -> None:
        """Test setting custom TTL."""
        node = MockNode()
        key = await store.store(node, "test-session")

        # Set custom TTL
        result = await store.set_ttl(key, 60)
        assert result is True

        ttl = await store.get_ttl(key)
        assert ttl is not None
        assert ttl <= 60

    async def test_get_ttl_missing_key(self, store: RedisStore) -> None:
        """Test get_ttl for missing key."""
        missing_key = StorageKey(session_id="test-missing", node_id=uuid4())
        ttl = await store.get_ttl(missing_key)
        assert ttl is None


@pytest.mark.asyncio
class TestRedisStoreClose:
    """Tests for close and lifecycle operations."""

    async def test_close_is_safe(self) -> None:
        """Test that close can be called safely."""
        store = RedisStore(REDIS_URL, key_prefix="test_close")
        await store.initialize()
        await store.close()
        await store.close()  # Should not raise

    async def test_operations_after_close_reinitialize(self) -> None:
        """Test that store can be reused after close."""
        store = RedisStore(REDIS_URL, key_prefix="test_reopen")
        await store.initialize()
        await store.close()

        # Should auto-reinitialize
        node = MockNode()
        key = await store.store(node, "test-reopen")
        assert key is not None

        await store.delete(key)
        await store.close()


@pytest.mark.asyncio
class TestRedisStoreIntegration:
    """Integration tests for full workflows."""

    async def test_full_lifecycle(self, store: RedisStore) -> None:
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
        results = await store.search_by_metadata("test-integration", min_importance=0.9)
        assert len(results) == 1

        # Stats
        stats = await store.stats("test-integration")
        assert stats.total_items == 1

        # Delete
        deleted = await store.delete(key)
        assert deleted is True
        assert await store.exists(key) is False

    async def test_session_isolation(self, store: RedisStore) -> None:
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

    async def test_flush_session(self, store: RedisStore) -> None:
        """Test flushing all keys for a session."""
        nodes = [MockNode() for _ in range(5)]
        await store.store_batch(nodes, "test-flush")

        keys_before = await store.list_keys("test-flush")
        assert len(keys_before) == 5

        deleted = await store.flush_session("test-flush")
        assert deleted == 5

        keys_after = await store.list_keys("test-flush")
        assert len(keys_after) == 0

    async def test_concurrent_access(self, store: RedisStore) -> None:
        """Test concurrent store operations."""
        import asyncio

        async def store_node(i: int) -> StorageKey:
            node = MockNode(content=f"Concurrent {i}")
            return await store.store(node, "test-concurrent")

        keys = await asyncio.gather(*[store_node(i) for i in range(10)])
        assert len(keys) == 10
        assert len({k.node_id for k in keys}) == 10  # All unique
