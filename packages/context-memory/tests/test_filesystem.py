"""Unit tests for FileSystemStore backend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from context_memory.backends.filesystem import FileSystemStore
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


class TestFileSystemStoreInit:
    """Tests for FileSystemStore initialization."""

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Test that base directory is created if it doesn't exist."""
        store_path = tmp_path / "new_store"
        assert not store_path.exists()

        store = FileSystemStore(store_path, create_if_missing=True)
        assert store_path.exists()
        assert store.base_path == store_path

    def test_raises_if_missing_and_create_disabled(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised if create_if_missing=False."""
        store_path = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            FileSystemStore(store_path, create_if_missing=False)

    def test_uses_existing_directory(self, tmp_path: Path) -> None:
        """Test that existing directory is used without error."""
        store_path = tmp_path / "existing"
        store_path.mkdir()

        store = FileSystemStore(store_path)
        assert store.base_path == store_path

    def test_satisfies_memory_store_protocol(self, tmp_path: Path) -> None:
        """Test that FileSystemStore satisfies MemoryStore protocol."""
        store = FileSystemStore(tmp_path)
        assert isinstance(store, MemoryStore)


class TestFileSystemStorePaths:
    """Tests for path helper methods."""

    def test_session_hash_is_consistent(self, tmp_path: Path) -> None:
        """Test that session hash is deterministic."""
        store = FileSystemStore(tmp_path)

        hash1 = store._session_hash("test-session")
        hash2 = store._session_hash("test-session")
        assert hash1 == hash2
        assert len(hash1) == 16

    def test_session_hash_differs_for_different_sessions(self, tmp_path: Path) -> None:
        """Test that different sessions get different hashes."""
        store = FileSystemStore(tmp_path)

        hash1 = store._session_hash("session-1")
        hash2 = store._session_hash("session-2")
        assert hash1 != hash2

    def test_node_path_format(self, tmp_path: Path) -> None:
        """Test that node path follows expected format."""
        store = FileSystemStore(tmp_path)
        key = StorageKey(session_id="sess", node_id=uuid4(), version=2)

        path = store._node_path(key)
        assert path.suffix == ".json"
        assert f".v{key.version}" in path.stem
        assert str(key.node_id) in path.stem

    def test_metadata_path_format(self, tmp_path: Path) -> None:
        """Test that metadata path follows expected format."""
        store = FileSystemStore(tmp_path)
        key = StorageKey(session_id="sess", node_id=uuid4(), version=1)

        path = store._metadata_path(key)
        assert path.suffix == ".json"
        assert ".meta" in path.stem


@pytest.mark.asyncio
class TestFileSystemStoreOperations:
    """Tests for FileSystemStore async operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileSystemStore:
        """Create a FileSystemStore for testing."""
        return FileSystemStore(tmp_path)

    async def test_store_creates_files(
        self, store: FileSystemStore, tmp_path: Path
    ) -> None:
        """Test that store creates node and metadata files."""
        node = MockNode()
        key = await store.store(node, "test-session")

        # Verify files exist
        node_path = store._node_path(key)
        metadata_path = store._metadata_path(key)

        assert node_path.exists()
        assert metadata_path.exists()

    async def test_store_with_custom_metadata(self, store: FileSystemStore) -> None:
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
        retrieved_meta = await store.get_metadata(key)

        assert retrieved_meta is not None
        assert retrieved_meta.tier == StorageTier.COLD
        assert retrieved_meta.importance == 0.9
        assert "custom" in retrieved_meta.tags

    async def test_store_batch(self, store: FileSystemStore) -> None:
        """Test batch store operation."""
        nodes = [MockNode(content=f"Node {i}") for i in range(5)]

        keys = await store.store_batch(nodes, "test-session")

        assert len(keys) == 5
        assert all(k.session_id == "test-session" for k in keys)

        # Verify all nodes exist
        for key in keys:
            assert await store.exists(key)

    async def test_exists_returns_true_for_stored_node(
        self, store: FileSystemStore
    ) -> None:
        """Test that exists returns True for stored nodes."""
        node = MockNode()
        key = await store.store(node, "test-session")

        assert await store.exists(key)

    async def test_exists_returns_false_for_missing_node(
        self, store: FileSystemStore
    ) -> None:
        """Test that exists returns False for non-existent nodes."""
        key = StorageKey(session_id="test", node_id=uuid4())
        assert not await store.exists(key)

    async def test_delete_removes_files(self, store: FileSystemStore) -> None:
        """Test that delete removes node and metadata files."""
        node = MockNode()
        key = await store.store(node, "test-session")

        node_path = store._node_path(key)
        metadata_path = store._metadata_path(key)

        assert node_path.exists()
        assert metadata_path.exists()

        deleted = await store.delete(key)

        assert deleted is True
        assert not node_path.exists()
        assert not metadata_path.exists()

    async def test_delete_returns_false_for_missing(
        self, store: FileSystemStore
    ) -> None:
        """Test that delete returns False for non-existent nodes."""
        key = StorageKey(session_id="test", node_id=uuid4())
        deleted = await store.delete(key)
        assert deleted is False

    async def test_get_metadata_returns_metadata(self, store: FileSystemStore) -> None:
        """Test that get_metadata returns stored metadata."""
        node = MockNode(token_count=42, importance=0.7)
        key = await store.store(node, "test-session")

        metadata = await store.get_metadata(key)

        assert metadata is not None
        assert metadata.token_count == 42
        assert metadata.tier == StorageTier.WARM

    async def test_get_metadata_returns_none_for_missing(
        self, store: FileSystemStore
    ) -> None:
        """Test that get_metadata returns None for missing keys."""
        key = StorageKey(session_id="test", node_id=uuid4())
        metadata = await store.get_metadata(key)
        assert metadata is None

    async def test_update_metadata(self, store: FileSystemStore) -> None:
        """Test updating metadata fields."""
        node = MockNode()
        key = await store.store(node, "test-session")

        # Update importance
        updated = await store.update_metadata(key, {"importance": 0.95})
        assert updated is True

        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.95

    async def test_update_metadata_returns_false_for_missing(
        self, store: FileSystemStore
    ) -> None:
        """Test that update_metadata returns False for missing keys."""
        key = StorageKey(session_id="test", node_id=uuid4())
        updated = await store.update_metadata(key, {"importance": 0.5})
        assert updated is False


@pytest.mark.asyncio
class TestFileSystemStoreQueries:
    """Tests for query operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileSystemStore:
        """Create a FileSystemStore for testing."""
        return FileSystemStore(tmp_path)

    async def test_list_keys_empty_session(self, store: FileSystemStore) -> None:
        """Test list_keys returns empty list for empty session."""
        keys = await store.list_keys("empty-session")
        assert keys == []

    async def test_list_keys_returns_all_keys(self, store: FileSystemStore) -> None:
        """Test list_keys returns all stored keys."""
        nodes = [MockNode() for _ in range(5)]
        stored_keys = await store.store_batch(nodes, "test-session")

        listed_keys = await store.list_keys("test-session")

        assert len(listed_keys) == 5
        # Check all stored keys are in listed keys (order may differ)
        stored_ids = {k.node_id for k in stored_keys}
        listed_ids = {k.node_id for k in listed_keys}
        assert stored_ids == listed_ids

    async def test_list_keys_respects_limit(self, store: FileSystemStore) -> None:
        """Test list_keys respects limit parameter."""
        nodes = [MockNode() for _ in range(10)]
        await store.store_batch(nodes, "test-session")

        keys = await store.list_keys("test-session", limit=5)
        assert len(keys) == 5

    async def test_list_keys_filters_by_node_type(self, store: FileSystemStore) -> None:
        """Test list_keys filters by node_type."""
        # Store nodes with different types
        for _ in range(3):
            node = MockNode(node_type="MESSAGE")
            await store.store(node, "test-session")

        for _ in range(2):
            node = MockNode(node_type="TOOL_CALL")
            await store.store(node, "test-session")

        message_keys = await store.list_keys("test-session", node_type="MESSAGE")
        assert len(message_keys) == 3

    async def test_search_by_metadata_importance(self, store: FileSystemStore) -> None:
        """Test search_by_metadata filters by importance."""
        # Store nodes with different importance
        for i in range(5):
            node = MockNode(importance=i * 0.2)
            await store.store(node, "test-session")

        # Search for high importance nodes
        results = await store.search_by_metadata("test-session", min_importance=0.5)

        assert len(results) >= 2  # 0.6 and 0.8 should match
        for _key, meta in results:
            assert meta.importance >= 0.5

    async def test_search_by_metadata_tags(self, store: FileSystemStore) -> None:
        """Test search_by_metadata filters by tags."""
        # Store node with tags
        node = MockNode(tags={"important", "review"})
        key = await store.store(node, "test-session")

        # Update metadata to have proper tags
        await store.update_metadata(key, {"tags": ["important", "review"]})

        # Store node without matching tags
        node2 = MockNode(tags={"other"})
        key2 = await store.store(node2, "test-session")
        await store.update_metadata(key2, {"tags": ["other"]})

        # Search for nodes with "important" tag
        results = await store.search_by_metadata("test-session", tags={"important"})

        assert len(results) >= 1

    async def test_search_by_metadata_since(self, store: FileSystemStore) -> None:
        """Test search_by_metadata filters by time."""
        node = MockNode()
        await store.store(node, "test-session")

        # Search since future time should return nothing
        future = datetime.now(UTC) + timedelta(hours=1)
        results = await store.search_by_metadata("test-session", since=future)
        assert len(results) == 0

        # Search since past should return the node
        past = datetime.now(UTC) - timedelta(hours=1)
        results = await store.search_by_metadata("test-session", since=past)
        assert len(results) == 1


@pytest.mark.asyncio
class TestFileSystemStoreStats:
    """Tests for statistics operations."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileSystemStore:
        """Create a FileSystemStore for testing."""
        return FileSystemStore(tmp_path)

    async def test_stats_empty_store(self, store: FileSystemStore) -> None:
        """Test stats on empty store."""
        stats = await store.stats("empty-session")

        assert stats.total_items == 0
        assert stats.total_size_bytes == 0
        assert stats.total_tokens == 0

    async def test_stats_with_data(self, store: FileSystemStore) -> None:
        """Test stats with stored data."""
        for _ in range(5):
            node = MockNode(token_count=100)
            await store.store(node, "test-session")

        stats = await store.stats("test-session")

        assert stats.total_items == 5
        assert stats.total_tokens == 500
        assert stats.total_size_bytes > 0

    async def test_stats_tier_breakdown(self, store: FileSystemStore) -> None:
        """Test stats includes tier breakdown."""
        node = MockNode()
        await store.store(node, "test-session")

        stats = await store.stats("test-session")

        # FileSystemStore defaults to WARM tier
        assert "warm" in stats.items_by_tier
        assert stats.items_by_tier["warm"] == 1

    async def test_stats_temporal_bounds(self, store: FileSystemStore) -> None:
        """Test stats includes temporal bounds."""
        for _ in range(3):
            node = MockNode()
            await store.store(node, "test-session")

        stats = await store.stats("test-session")

        assert stats.oldest_item is not None
        assert stats.newest_item is not None
        assert stats.oldest_item <= stats.newest_item


@pytest.mark.asyncio
class TestFileSystemStoreClose:
    """Tests for close operation."""

    async def test_close_is_safe(self, tmp_path: Path) -> None:
        """Test that close can be called multiple times safely."""
        store = FileSystemStore(tmp_path)

        await store.close()
        await store.close()  # Should not raise


@pytest.mark.asyncio
class TestFileSystemStoreIntegration:
    """Integration tests for FileSystemStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> FileSystemStore:
        """Create a FileSystemStore for testing."""
        return FileSystemStore(tmp_path)

    async def test_full_lifecycle(self, store: FileSystemStore) -> None:
        """Test complete store -> retrieve -> update -> delete lifecycle."""
        # Store
        node = MockNode(content="Test content", token_count=50)
        key = await store.store(node, "lifecycle-session")

        # Verify stored
        assert await store.exists(key)

        # Get metadata
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.token_count == 50
        assert metadata.access_count == 0

        # Update metadata
        await store.update_metadata(key, {"importance": 0.9})
        metadata = await store.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.9

        # List keys
        keys = await store.list_keys("lifecycle-session")
        assert len(keys) == 1
        assert keys[0].node_id == key.node_id

        # Get stats
        stats = await store.stats("lifecycle-session")
        assert stats.total_items == 1

        # Delete
        deleted = await store.delete(key)
        assert deleted is True
        assert not await store.exists(key)

        # Verify stats updated
        stats = await store.stats("lifecycle-session")
        assert stats.total_items == 0

    async def test_multi_session_isolation(self, store: FileSystemStore) -> None:
        """Test that sessions are isolated from each other."""
        # Store in different sessions
        node1 = MockNode()
        node2 = MockNode()

        await store.store(node1, "session-1")
        await store.store(node2, "session-2")

        # List keys should be isolated
        keys1 = await store.list_keys("session-1")
        keys2 = await store.list_keys("session-2")

        assert len(keys1) == 1
        assert len(keys2) == 1
        assert keys1[0].node_id != keys2[0].node_id

        # Stats should be isolated
        stats1 = await store.stats("session-1")
        stats2 = await store.stats("session-2")

        assert stats1.total_items == 1
        assert stats2.total_items == 1
