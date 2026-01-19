"""Unit tests for WorkingMemory."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from context_memory.types import StorageKey, StorageMetadata, StorageTier
from context_memory.working import WorkingMemory, WorkingMemoryStats

# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockNodeMetadata:
    """Mock metadata for context nodes."""

    importance: float = 0.5
    tags: set[str] = field(default_factory=set)


@dataclass
class MockContextNode:
    """Mock context node for testing."""

    id: UUID = field(default_factory=uuid4)
    type: str = "MESSAGE"
    token_count: int = 100
    metadata: MockNodeMetadata = field(default_factory=MockNodeMetadata)


class MockMemoryStore:
    """Mock memory store for testing."""

    def __init__(self) -> None:
        self.stored: dict[UUID, tuple[Any, str, StorageMetadata]] = {}
        self.store_calls: int = 0
        self.retrieve_calls: int = 0

    async def store(
        self,
        node: Any,
        session_id: str,
        metadata: StorageMetadata,
    ) -> None:
        """Store a node."""
        self.stored[node.id] = (node, session_id, metadata)
        self.store_calls += 1

    async def retrieve(self, key: StorageKey) -> Any | None:
        """Retrieve a node by key."""
        self.retrieve_calls += 1
        for node_id, (node, session_id, _) in self.stored.items():
            if node_id == key.node_id and session_id == key.session_id:
                return node
        return None

    async def list_keys(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[StorageKey]:
        """List keys for a session."""
        keys: list[StorageKey] = []
        for _node_id, (_, stored_session, metadata) in self.stored.items():
            if stored_session == session_id and len(keys) < limit:
                keys.append(metadata.key)
        return keys

    async def get_metadata(self, key: StorageKey) -> StorageMetadata | None:
        """Get metadata for a key."""
        for node_id, (_, session_id, metadata) in self.stored.items():
            if node_id == key.node_id and session_id == key.session_id:
                return metadata
        return None


# =============================================================================
# WorkingMemoryStats Tests
# =============================================================================


class TestWorkingMemoryStats:
    """Tests for WorkingMemoryStats."""

    def test_create_stats(self) -> None:
        """Test creating stats object."""
        stats = WorkingMemoryStats(
            items=10,
            tokens=5000,
            max_tokens=10000,
            max_items=100,
            dirty_items=3,
        )
        assert stats.items == 10
        assert stats.tokens == 5000
        assert stats.max_tokens == 10000
        assert stats.max_items == 100
        assert stats.dirty_items == 3

    def test_token_utilization(self) -> None:
        """Test token utilization calculation."""
        stats = WorkingMemoryStats(
            items=10,
            tokens=5000,
            max_tokens=10000,
            max_items=100,
            dirty_items=0,
        )
        assert stats.token_utilization == 0.5

    def test_item_utilization(self) -> None:
        """Test item utilization calculation."""
        stats = WorkingMemoryStats(
            items=25,
            tokens=5000,
            max_tokens=10000,
            max_items=100,
            dirty_items=0,
        )
        assert stats.item_utilization == 0.25

    def test_zero_max_tokens(self) -> None:
        """Test utilization with zero max tokens."""
        stats = WorkingMemoryStats(
            items=10,
            tokens=5000,
            max_tokens=0,
            max_items=100,
            dirty_items=0,
        )
        assert stats.token_utilization == 0.0

    def test_zero_max_items(self) -> None:
        """Test utilization with zero max items."""
        stats = WorkingMemoryStats(
            items=10,
            tokens=5000,
            max_tokens=10000,
            max_items=0,
            dirty_items=0,
        )
        assert stats.item_utilization == 0.0

    def test_to_dict(self) -> None:
        """Test converting stats to dictionary."""
        stats = WorkingMemoryStats(
            items=10,
            tokens=5000,
            max_tokens=10000,
            max_items=100,
            dirty_items=3,
        )
        result = stats.to_dict()
        assert result["items"] == 10
        assert result["tokens"] == 5000
        assert result["token_utilization"] == 0.5
        assert result["item_utilization"] == 0.1
        assert result["dirty_items"] == 3


# =============================================================================
# WorkingMemory Tests
# =============================================================================


class TestWorkingMemory:
    """Tests for WorkingMemory."""

    @pytest.fixture
    def mock_store(self) -> MockMemoryStore:
        """Create mock store."""
        return MockMemoryStore()

    @pytest.fixture
    def memory(self, mock_store: MockMemoryStore) -> WorkingMemory:
        """Create working memory with mock store."""
        return WorkingMemory(
            backing_store=mock_store,
            max_tokens=1000,
            max_items=10,
            sync_interval_seconds=3600,  # Long interval for tests
        )

    # -------------------------------------------------------------------------
    # Lifecycle Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_creates_sync_task(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test start creates background sync task."""
        await memory.start()
        try:
            assert memory._sync_task is not None
            assert not memory._sync_task.done()
        finally:
            await memory.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test multiple start calls don't create multiple tasks."""
        await memory.start()
        try:
            task1 = memory._sync_task
            await memory.start()
            task2 = memory._sync_task
            assert task1 is task2
        finally:
            await memory.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test stop cancels sync task."""
        await memory.start()
        await memory.stop()
        assert memory._sync_task is None
        assert memory._closed is True

    @pytest.mark.asyncio
    async def test_context_manager(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test async context manager."""
        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=1000,
            max_items=10,
        )
        async with memory:
            assert memory._sync_task is not None
        assert memory._sync_task is None
        assert memory._closed is True

    # -------------------------------------------------------------------------
    # Core Operations Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_add_node(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test adding a node to cache."""
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test-session")

        assert memory.item_count == 1
        assert memory.token_count == 100
        assert memory.dirty_count == 1

    @pytest.mark.asyncio
    async def test_add_multiple_nodes(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test adding multiple nodes."""
        for _ in range(5):
            node = MockContextNode(token_count=50)
            await memory.add(node, session_id="test-session")

        assert memory.item_count == 5
        assert memory.token_count == 250
        assert memory.dirty_count == 5

    @pytest.mark.asyncio
    async def test_get_node(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test getting a node from cache."""
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test-session")

        result = await memory.get(node.id)
        assert result is not None
        assert result.id == node.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_node(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test getting nonexistent node returns None."""
        result = await memory.get(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_updates_lru_order(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test get moves node to end (most recently used)."""
        node1 = MockContextNode(token_count=100)
        node2 = MockContextNode(token_count=100)

        await memory.add(node1, session_id="test")
        await memory.add(node2, session_id="test")

        # node1 is LRU, node2 is MRU
        # Access node1 to make it MRU
        await memory.get(node1.id)

        # Now node2 should be LRU
        lru_key = next(iter(memory._cache.keys()))
        assert lru_key == node2.id

    @pytest.mark.asyncio
    async def test_remove_node(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test removing a node from cache."""
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test-session")

        result = await memory.remove(node.id)
        assert result is not None
        assert result.id == node.id
        assert memory.item_count == 0
        assert memory.token_count == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_node(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test removing nonexistent node returns None."""
        result = await memory.remove(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_contains(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test contains check."""
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test-session")

        assert await memory.contains(node.id) is True
        assert await memory.contains(uuid4()) is False

    @pytest.mark.asyncio
    async def test_clear(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test clearing all items."""
        for _ in range(5):
            node = MockContextNode(token_count=50)
            await memory.add(node, session_id="test-session")

        count = await memory.clear()
        assert count == 5
        assert memory.item_count == 0
        assert memory.token_count == 0
        assert memory.dirty_count == 0

    # -------------------------------------------------------------------------
    # Eviction Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_eviction_on_token_limit(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test eviction when token limit exceeded."""
        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=300,  # Only room for 3 nodes at 100 tokens each
            max_items=100,
        )

        nodes: list[MockContextNode] = []
        for _ in range(5):
            node = MockContextNode(token_count=100)
            nodes.append(node)
            await memory.add(node, session_id="test")

        # Should have evicted 2 oldest nodes
        assert memory.item_count == 3
        assert memory.token_count == 300

        # First 2 nodes should be evicted (persisted to store)
        assert mock_store.store_calls == 2

    @pytest.mark.asyncio
    async def test_eviction_on_item_limit(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test eviction when item limit exceeded."""
        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=10000,
            max_items=3,  # Only room for 3 items
        )

        nodes: list[MockContextNode] = []
        for _ in range(5):
            node = MockContextNode(token_count=10)
            nodes.append(node)
            await memory.add(node, session_id="test")

        # Should have evicted 2 oldest nodes
        assert memory.item_count == 3

        # First 2 nodes should be evicted (persisted to store)
        assert mock_store.store_calls == 2

    @pytest.mark.asyncio
    async def test_eviction_persists_dirty_items(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test eviction persists dirty items before removal."""
        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=200,  # Room for 2 nodes
            max_items=100,
        )

        node1 = MockContextNode(token_count=100)
        node2 = MockContextNode(token_count=100)
        node3 = MockContextNode(token_count=100)

        await memory.add(node1, session_id="test")
        await memory.add(node2, session_id="test")

        # Adding node3 should evict node1
        await memory.add(node3, session_id="test")

        # node1 should be in the store
        assert node1.id in mock_store.stored

    # -------------------------------------------------------------------------
    # Persistence Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_flush_persists_dirty_items(
        self,
        memory: WorkingMemory,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test flush persists all dirty items."""
        for _ in range(3):
            node = MockContextNode(token_count=50)
            await memory.add(node, session_id="test")

        assert memory.dirty_count == 3

        flushed = await memory.flush()
        assert flushed == 3
        assert memory.dirty_count == 0
        assert mock_store.store_calls == 3

    @pytest.mark.asyncio
    async def test_flush_clears_dirty_set(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test flush clears dirty tracking."""
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test")

        await memory.flush()
        assert memory.dirty_count == 0

        # Second flush should do nothing
        flushed = await memory.flush()
        assert flushed == 0

    @pytest.mark.asyncio
    async def test_stop_flushes_dirty_items(
        self,
        memory: WorkingMemory,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test stop flushes all dirty items."""
        await memory.start()

        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test")

        await memory.stop()

        # Should have been flushed on stop
        assert mock_store.store_calls == 1

    # -------------------------------------------------------------------------
    # Load from Store Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_load_from_store(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test loading items from backing store."""
        # Pre-populate store
        for _ in range(3):
            node = MockContextNode(token_count=50)
            key = StorageKey(session_id="test", node_id=node.id, version=1)
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.HOT,
                size_bytes=0,
                token_count=50,
                node_type="MESSAGE",
                importance=0.5,
                tags=set(),
            )
            mock_store.stored[node.id] = (node, "test", metadata)

        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=1000,
            max_items=100,
        )

        loaded = await memory.load_from_store("test", limit=10)
        assert loaded == 3
        assert memory.item_count == 3

    @pytest.mark.asyncio
    async def test_load_from_store_respects_token_limit(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test load respects token capacity."""
        # Pre-populate store with nodes
        for _ in range(5):
            node = MockContextNode(token_count=100)
            key = StorageKey(session_id="test", node_id=node.id, version=1)
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.HOT,
                size_bytes=0,
                token_count=100,
                node_type="MESSAGE",
                importance=0.5,
                tags=set(),
            )
            mock_store.stored[node.id] = (node, "test", metadata)

        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=200,  # Only room for 2 nodes
            max_items=100,
        )

        loaded = await memory.load_from_store("test", limit=10)
        assert loaded == 2
        assert memory.token_count <= 200

    @pytest.mark.asyncio
    async def test_load_from_store_respects_item_limit(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test load respects item capacity."""
        # Pre-populate store
        for _ in range(5):
            node = MockContextNode(token_count=10)
            key = StorageKey(session_id="test", node_id=node.id, version=1)
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.HOT,
                size_bytes=0,
                token_count=10,
                node_type="MESSAGE",
                importance=0.5,
                tags=set(),
            )
            mock_store.stored[node.id] = (node, "test", metadata)

        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=10000,
            max_items=3,  # Only room for 3 items
        )

        loaded = await memory.load_from_store("test", limit=10)
        assert loaded == 3
        assert memory.item_count == 3

    @pytest.mark.asyncio
    async def test_load_skips_already_cached(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test load skips items already in cache."""
        # Create a node and add to both store and memory
        node = MockContextNode(token_count=100)
        key = StorageKey(session_id="test", node_id=node.id, version=1)
        metadata = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=0,
            token_count=100,
            node_type="MESSAGE",
            importance=0.5,
            tags=set(),
        )
        mock_store.stored[node.id] = (node, "test", metadata)

        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=1000,
            max_items=100,
        )

        # Add to cache first
        await memory.add(node, session_id="test")

        # Load should skip since already cached
        loaded = await memory.load_from_store("test", limit=10)
        assert loaded == 0

    @pytest.mark.asyncio
    async def test_loaded_items_not_dirty(
        self,
        mock_store: MockMemoryStore,
    ) -> None:
        """Test loaded items are not marked dirty."""
        node = MockContextNode(token_count=100)
        key = StorageKey(session_id="test", node_id=node.id, version=1)
        metadata = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=0,
            token_count=100,
            node_type="MESSAGE",
            importance=0.5,
            tags=set(),
        )
        mock_store.stored[node.id] = (node, "test", metadata)

        memory = WorkingMemory(
            backing_store=mock_store,
            max_tokens=1000,
            max_items=100,
        )

        await memory.load_from_store("test", limit=10)
        assert memory.dirty_count == 0

    # -------------------------------------------------------------------------
    # Statistics Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stats_property(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test stats property returns correct values."""
        for _ in range(3):
            node = MockContextNode(token_count=100)
            await memory.add(node, session_id="test")

        stats = memory.stats
        assert stats.items == 3
        assert stats.tokens == 300
        assert stats.max_tokens == 1000
        assert stats.max_items == 10
        assert stats.dirty_items == 3

    @pytest.mark.asyncio
    async def test_item_count_property(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test item_count property."""
        assert memory.item_count == 0
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test")
        assert memory.item_count == 1

    @pytest.mark.asyncio
    async def test_token_count_property(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test token_count property."""
        assert memory.token_count == 0
        node = MockContextNode(token_count=150)
        await memory.add(node, session_id="test")
        assert memory.token_count == 150

    @pytest.mark.asyncio
    async def test_dirty_count_property(
        self,
        memory: WorkingMemory,
    ) -> None:
        """Test dirty_count property."""
        assert memory.dirty_count == 0
        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test")
        assert memory.dirty_count == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestWorkingMemoryEdgeCases:
    """Edge case tests for WorkingMemory."""

    @pytest.mark.asyncio
    async def test_add_node_with_zero_tokens(self) -> None:
        """Test adding node with zero token count."""
        store = MockMemoryStore()
        memory = WorkingMemory(
            backing_store=store,
            max_tokens=1000,
            max_items=10,
        )

        node = MockContextNode(token_count=0)
        await memory.add(node, session_id="test")

        assert memory.item_count == 1
        assert memory.token_count == 0

    @pytest.mark.asyncio
    async def test_add_node_with_missing_token_count(self) -> None:
        """Test adding node without token_count attribute."""
        store = MockMemoryStore()
        memory = WorkingMemory(
            backing_store=store,
            max_tokens=1000,
            max_items=10,
        )

        @dataclass
        class NodeWithoutTokens:
            id: UUID = field(default_factory=uuid4)
            type: str = "MESSAGE"

        node = NodeWithoutTokens()
        await memory.add(node, session_id="test")

        assert memory.item_count == 1
        assert memory.token_count == 0

    @pytest.mark.asyncio
    async def test_eviction_from_empty_cache(self) -> None:
        """Test eviction from empty cache does nothing."""
        store = MockMemoryStore()
        memory = WorkingMemory(
            backing_store=store,
            max_tokens=1000,
            max_items=10,
        )

        # Manually call evict on empty cache
        await memory._evict_one()
        assert memory.item_count == 0

    @pytest.mark.asyncio
    async def test_get_updates_metadata_access_time(self) -> None:
        """Test get updates metadata access tracking."""
        store = MockMemoryStore()
        memory = WorkingMemory(
            backing_store=store,
            max_tokens=1000,
            max_items=10,
        )

        node = MockContextNode(token_count=100)
        await memory.add(node, session_id="test")

        original_access_count = memory._metadata[node.id].access_count

        await memory.get(node.id)

        assert memory._metadata[node.id].access_count == original_access_count + 1

    @pytest.mark.asyncio
    async def test_concurrent_operations(self) -> None:
        """Test concurrent add/get operations."""
        store = MockMemoryStore()
        memory = WorkingMemory(
            backing_store=store,
            max_tokens=10000,
            max_items=100,
        )

        async def add_and_get() -> None:
            node = MockContextNode(token_count=10)
            await memory.add(node, session_id="test")
            await memory.get(node.id)

        # Run multiple concurrent operations
        await asyncio.gather(*[add_and_get() for _ in range(10)])

        assert memory.item_count == 10
