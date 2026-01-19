"""Unit tests for TieredStorage coordinator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from context_memory.backends.filesystem import FileSystemStore
from context_memory.tiered import MigrationResult, TierConfig, TieredStorage
from context_memory.types import (
    StorageKey,
    StorageTier,
)


class MockNodeMetadata:
    """Mock node metadata for testing."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNode:
    """Mock ContextNode for testing without importing context-core.

    This mock produces serialized output compatible with the real ContextNode
    so that deserialization works correctly in tests.
    """

    def __init__(
        self,
        node_type: str = "message",  # lowercase to match NodeType enum
        content: str = "test content",
        token_count: int = 10,
        importance: float = 0.5,
        tags: set[str] | None = None,
    ) -> None:
        self.id = uuid4()
        # Use lowercase to match NodeType enum values
        self.type = type("NodeType", (), {"value": node_type.lower()})()
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance, tags)

    def model_dump(self, mode: str = "python") -> dict:
        """Serialize node to dict compatible with ContextNode."""
        return {
            "id": str(self.id),
            "type": self.type.value,  # Already lowercase
            # Content must be a dict with text field for ContextNode
            "content": {"text": self.content, "role": "user"},
            "token_count": self.token_count,
            "metadata": {
                "importance": self.metadata.importance,
                "tags": list(self.metadata.tags),
            },
        }


# =============================================================================
# TierConfig Tests
# =============================================================================


class TestTierConfig:
    """Tests for TierConfig dataclass."""

    def test_basic_config(self, tmp_path: Path) -> None:
        """Test basic TierConfig creation."""
        backend = FileSystemStore(tmp_path)
        config = TierConfig(
            tier=StorageTier.HOT,
            backend=backend,
            max_age_seconds=3600,
            max_items=1000,
            min_importance=0.3,
        )

        assert config.tier == StorageTier.HOT
        assert config.backend is backend
        assert config.max_age_seconds == 3600
        assert config.max_items == 1000
        assert config.min_importance == 0.3
        assert config.promote_on_access is True  # default

    def test_config_defaults(self, tmp_path: Path) -> None:
        """Test TierConfig default values."""
        backend = FileSystemStore(tmp_path)
        config = TierConfig(tier=StorageTier.WARM, backend=backend)

        assert config.max_age_seconds is None
        assert config.max_items is None
        assert config.min_importance == 0.0
        assert config.promote_on_access is True


# =============================================================================
# TieredStorage Initialization Tests
# =============================================================================


class TestTieredStorageInit:
    """Tests for TieredStorage initialization."""

    def test_single_tier(self, tmp_path: Path) -> None:
        """Test creating TieredStorage with single tier."""
        backend = FileSystemStore(tmp_path / "warm")
        config = TierConfig(tier=StorageTier.WARM, backend=backend)

        tiered = TieredStorage([config])

        assert StorageTier.WARM in tiered._tiers
        assert len(tiered._tiers) == 1

    def test_multiple_tiers(self, tmp_path: Path) -> None:
        """Test creating TieredStorage with multiple tiers."""
        hot_backend = FileSystemStore(tmp_path / "hot")
        warm_backend = FileSystemStore(tmp_path / "warm")
        cold_backend = FileSystemStore(tmp_path / "cold")

        tiered = TieredStorage(
            [
                TierConfig(StorageTier.HOT, hot_backend),
                TierConfig(StorageTier.WARM, warm_backend),
                TierConfig(StorageTier.COLD, cold_backend),
            ]
        )

        assert len(tiered._tiers) == 3
        assert StorageTier.HOT in tiered._tiers
        assert StorageTier.WARM in tiered._tiers
        assert StorageTier.COLD in tiered._tiers

    def test_duplicate_tiers_raises(self, tmp_path: Path) -> None:
        """Test that duplicate tier configurations raise ValueError."""
        backend1 = FileSystemStore(tmp_path / "hot1")
        backend2 = FileSystemStore(tmp_path / "hot2")

        with pytest.raises(ValueError, match="Duplicate tier"):
            TieredStorage(
                [
                    TierConfig(StorageTier.HOT, backend1),
                    TierConfig(StorageTier.HOT, backend2),
                ]
            )

    def test_promotion_setting(self, tmp_path: Path) -> None:
        """Test promotion_on_access setting."""
        backend = FileSystemStore(tmp_path)

        tiered_default = TieredStorage([TierConfig(StorageTier.WARM, backend)])
        assert tiered_default._promotion_on_access is True

        tiered_disabled = TieredStorage(
            [TierConfig(StorageTier.WARM, backend)],
            promotion_on_access=False,
        )
        assert tiered_disabled._promotion_on_access is False

    def test_migration_interval_setting(self, tmp_path: Path) -> None:
        """Test migration_interval_seconds setting."""
        backend = FileSystemStore(tmp_path)

        tiered_default = TieredStorage([TierConfig(StorageTier.WARM, backend)])
        assert tiered_default._migration_interval == 300

        tiered_custom = TieredStorage(
            [TierConfig(StorageTier.WARM, backend)],
            migration_interval_seconds=60,
        )
        assert tiered_custom._migration_interval == 60


# =============================================================================
# Tier Navigation Tests
# =============================================================================


class TestTierNavigation:
    """Tests for tier navigation helpers."""

    @pytest.fixture
    def three_tier_storage(self, tmp_path: Path) -> TieredStorage:
        """Create a three-tier storage for testing."""
        return TieredStorage(
            [
                TierConfig(StorageTier.HOT, FileSystemStore(tmp_path / "hot")),
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
                TierConfig(StorageTier.COLD, FileSystemStore(tmp_path / "cold")),
            ]
        )

    def test_get_next_tier_from_hot(self, three_tier_storage: TieredStorage) -> None:
        """Test getting next tier from HOT."""
        assert three_tier_storage._get_next_tier(StorageTier.HOT) == StorageTier.WARM

    def test_get_next_tier_from_warm(self, three_tier_storage: TieredStorage) -> None:
        """Test getting next tier from WARM."""
        assert three_tier_storage._get_next_tier(StorageTier.WARM) == StorageTier.COLD

    def test_get_next_tier_from_cold(self, three_tier_storage: TieredStorage) -> None:
        """Test getting next tier from COLD returns None."""
        assert three_tier_storage._get_next_tier(StorageTier.COLD) is None

    def test_get_prev_tier_from_cold(self, three_tier_storage: TieredStorage) -> None:
        """Test getting previous tier from COLD."""
        assert three_tier_storage._get_prev_tier(StorageTier.COLD) == StorageTier.WARM

    def test_get_prev_tier_from_warm(self, three_tier_storage: TieredStorage) -> None:
        """Test getting previous tier from WARM."""
        assert three_tier_storage._get_prev_tier(StorageTier.WARM) == StorageTier.HOT

    def test_get_prev_tier_from_hot(self, three_tier_storage: TieredStorage) -> None:
        """Test getting previous tier from HOT returns None."""
        assert three_tier_storage._get_prev_tier(StorageTier.HOT) is None

    def test_get_hottest_tier(self, three_tier_storage: TieredStorage) -> None:
        """Test getting hottest configured tier."""
        assert three_tier_storage._get_hottest_tier() == StorageTier.HOT

    def test_get_hottest_tier_partial(self, tmp_path: Path) -> None:
        """Test hottest tier when only warm/cold configured."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
                TierConfig(StorageTier.COLD, FileSystemStore(tmp_path / "cold")),
            ]
        )
        assert tiered._get_hottest_tier() == StorageTier.WARM

    def test_get_configured_tiers_in_order(
        self, three_tier_storage: TieredStorage
    ) -> None:
        """Test getting configured tiers in hot-to-cold order."""
        tiers = three_tier_storage._get_configured_tiers_in_order()
        assert tiers == [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]


# =============================================================================
# Core Storage Operation Tests
# =============================================================================


class TestTieredStorageOperations:
    """Tests for TieredStorage CRUD operations."""

    @pytest_asyncio.fixture
    async def tiered_storage(self, tmp_path: Path) -> TieredStorage:
        """Create a tiered storage for testing."""
        storage = TieredStorage(
            [
                TierConfig(StorageTier.HOT, FileSystemStore(tmp_path / "hot")),
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
                TierConfig(StorageTier.COLD, FileSystemStore(tmp_path / "cold")),
            ]
        )
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_store_default_tier(self, tiered_storage: TieredStorage) -> None:
        """Test storing in default (hottest) tier."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1")

        assert key.session_id == "session-1"
        assert key.node_id == node.id

        # Verify stored in hot tier
        hot_backend = tiered_storage._tiers[StorageTier.HOT].backend
        assert await hot_backend.exists(key)

    @pytest.mark.asyncio
    async def test_store_specific_tier(self, tiered_storage: TieredStorage) -> None:
        """Test storing in a specific tier."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.COLD)

        # Verify stored only in cold tier
        cold_backend = tiered_storage._tiers[StorageTier.COLD].backend
        hot_backend = tiered_storage._tiers[StorageTier.HOT].backend

        assert await cold_backend.exists(key)
        assert not await hot_backend.exists(key)

    @pytest.mark.asyncio
    async def test_store_unconfigured_tier_raises(self, tmp_path: Path) -> None:
        """Test storing in unconfigured tier raises ValueError."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        )

        node = MockNode()
        with pytest.raises(ValueError, match="not configured"):
            await tiered.store(node, "session-1", tier=StorageTier.HOT)

        await tiered.close()

    @pytest.mark.asyncio
    async def test_retrieve_from_hot(self, tiered_storage: TieredStorage) -> None:
        """Test retrieving from hot tier."""
        node = MockNode(content="hot content")
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.HOT)

        retrieved = await tiered_storage.retrieve(key)

        assert retrieved is not None
        # Retrieved is a ContextNode with content.text
        assert retrieved.content.text == "hot content"

    @pytest.mark.asyncio
    async def test_retrieve_from_cold(self, tiered_storage: TieredStorage) -> None:
        """Test retrieving from cold tier."""
        node = MockNode(content="cold content")
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.COLD)

        retrieved = await tiered_storage.retrieve(key, promote=False)

        assert retrieved is not None
        # Retrieved is a ContextNode with content.text
        assert retrieved.content.text == "cold content"

    @pytest.mark.asyncio
    async def test_retrieve_promotes_from_cold_to_hot(
        self, tiered_storage: TieredStorage
    ) -> None:
        """Test that retrieve promotes items from cold to hot tier."""
        node = MockNode(content="promote me")
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.COLD)

        cold_backend = tiered_storage._tiers[StorageTier.COLD].backend
        hot_backend = tiered_storage._tiers[StorageTier.HOT].backend

        # Verify initially in cold only
        assert await cold_backend.exists(key)
        assert not await hot_backend.exists(key)

        # Retrieve with promotion (default)
        retrieved = await tiered_storage.retrieve(key, promote=True)
        assert retrieved is not None

        # Verify promoted to hot, removed from cold
        assert await hot_backend.exists(key)
        assert not await cold_backend.exists(key)

    @pytest.mark.asyncio
    async def test_retrieve_no_promotion(self, tiered_storage: TieredStorage) -> None:
        """Test retrieve without promotion."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.COLD)

        cold_backend = tiered_storage._tiers[StorageTier.COLD].backend
        hot_backend = tiered_storage._tiers[StorageTier.HOT].backend

        # Retrieve without promotion
        await tiered_storage.retrieve(key, promote=False)

        # Verify still in cold
        assert await cold_backend.exists(key)
        assert not await hot_backend.exists(key)

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, tiered_storage: TieredStorage) -> None:
        """Test retrieving nonexistent key returns None."""
        key = StorageKey(session_id="session-1", node_id=uuid4(), version=1)
        result = await tiered_storage.retrieve(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_from_single_tier(self, tiered_storage: TieredStorage) -> None:
        """Test deleting item from single tier."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.WARM)

        deleted = await tiered_storage.delete(key)
        assert deleted is True

        # Verify gone
        assert await tiered_storage.exists(key) is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tiered_storage: TieredStorage) -> None:
        """Test deleting nonexistent key returns False."""
        key = StorageKey(session_id="session-1", node_id=uuid4(), version=1)
        deleted = await tiered_storage.delete(key)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_exists(self, tiered_storage: TieredStorage) -> None:
        """Test exists check across tiers."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.COLD)

        assert await tiered_storage.exists(key) is True

        nonexistent = StorageKey(session_id="x", node_id=uuid4(), version=1)
        assert await tiered_storage.exists(nonexistent) is False

    @pytest.mark.asyncio
    async def test_store_batch(self, tiered_storage: TieredStorage) -> None:
        """Test batch storing."""
        nodes = [MockNode(content=f"node-{idx}") for idx in range(3)]
        keys = await tiered_storage.store_batch(nodes, "session-1")

        assert len(keys) == 3
        for key in keys:
            assert await tiered_storage.exists(key)

    @pytest.mark.asyncio
    async def test_retrieve_batch(self, tiered_storage: TieredStorage) -> None:
        """Test batch retrieval."""
        nodes = [MockNode(content=f"node-{idx}") for idx in range(3)]
        keys = await tiered_storage.store_batch(nodes, "session-1")

        results = await tiered_storage.retrieve_batch(keys, promote=False)

        assert len(results) == 3
        for result in results:
            assert result is not None


# =============================================================================
# Metadata Operation Tests
# =============================================================================


class TestTieredStorageMetadata:
    """Tests for metadata operations."""

    @pytest_asyncio.fixture
    async def tiered_storage(self, tmp_path: Path) -> TieredStorage:
        """Create a tiered storage for testing."""
        storage = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
            ]
        )
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_get_metadata(self, tiered_storage: TieredStorage) -> None:
        """Test getting metadata for stored item."""
        node = MockNode(importance=0.8, tags={"test", "metadata"})
        key = await tiered_storage.store(node, "session-1")

        metadata = await tiered_storage.get_metadata(key)

        assert metadata is not None
        assert metadata.key == key
        assert metadata.importance == 0.8

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(
        self, tiered_storage: TieredStorage
    ) -> None:
        """Test getting metadata for nonexistent item."""
        key = StorageKey(session_id="x", node_id=uuid4(), version=1)
        metadata = await tiered_storage.get_metadata(key)
        assert metadata is None

    @pytest.mark.asyncio
    async def test_update_metadata(self, tiered_storage: TieredStorage) -> None:
        """Test updating metadata."""
        node = MockNode(importance=0.5)
        key = await tiered_storage.store(node, "session-1")

        updated = await tiered_storage.update_metadata(key, {"importance": 0.9})
        assert updated is True

        metadata = await tiered_storage.get_metadata(key)
        assert metadata is not None
        assert metadata.importance == 0.9


# =============================================================================
# Migration Tests
# =============================================================================


class TestTieredStorageMigration:
    """Tests for tier migration functionality."""

    @pytest_asyncio.fixture
    async def tiered_storage(self, tmp_path: Path) -> TieredStorage:
        """Create a tiered storage for testing."""
        storage = TieredStorage(
            [
                TierConfig(
                    StorageTier.HOT,
                    FileSystemStore(tmp_path / "hot"),
                    max_age_seconds=1,  # 1 second for testing
                ),
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
                TierConfig(StorageTier.COLD, FileSystemStore(tmp_path / "cold")),
            ]
        )
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_migrate_to_tier(self, tiered_storage: TieredStorage) -> None:
        """Test manual migration to a specific tier."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.HOT)

        hot_backend = tiered_storage._tiers[StorageTier.HOT].backend
        cold_backend = tiered_storage._tiers[StorageTier.COLD].backend

        # Verify in hot
        assert await hot_backend.exists(key)

        # Migrate to cold
        result = await tiered_storage.migrate_to_tier(key, StorageTier.COLD)
        assert result is True

        # Verify moved
        assert not await hot_backend.exists(key)
        assert await cold_backend.exists(key)

    @pytest.mark.asyncio
    async def test_migrate_to_same_tier(self, tiered_storage: TieredStorage) -> None:
        """Test migration to same tier returns True."""
        node = MockNode()
        key = await tiered_storage.store(node, "session-1", tier=StorageTier.HOT)

        result = await tiered_storage.migrate_to_tier(key, StorageTier.HOT)
        assert result is True

    @pytest.mark.asyncio
    async def test_migrate_nonexistent(self, tiered_storage: TieredStorage) -> None:
        """Test migrating nonexistent key returns False."""
        key = StorageKey(session_id="x", node_id=uuid4(), version=1)
        result = await tiered_storage.migrate_to_tier(key, StorageTier.COLD)
        assert result is False

    @pytest.mark.asyncio
    async def test_migrate_to_unconfigured_tier_raises(self, tmp_path: Path) -> None:
        """Test migration to unconfigured tier raises ValueError."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        )

        node = MockNode()
        key = await tiered.store(node, "session-1")

        with pytest.raises(ValueError, match="not configured"):
            await tiered.migrate_to_tier(key, StorageTier.HOT)

        await tiered.close()


# =============================================================================
# Migration Task Tests
# =============================================================================


class TestMigrationTask:
    """Tests for background migration task."""

    @pytest.mark.asyncio
    async def test_start_stop_migration_task(self, tmp_path: Path) -> None:
        """Test starting and stopping migration task."""
        tiered = TieredStorage(
            [TierConfig(StorageTier.WARM, FileSystemStore(tmp_path))],
            migration_interval_seconds=1,
        )

        assert tiered._migration_task is None

        await tiered.start_migration_task()
        assert tiered._migration_task is not None
        assert not tiered._migration_task.done()

        await tiered.stop_migration_task()
        assert tiered._migration_task is None

        await tiered.close()

    @pytest.mark.asyncio
    async def test_run_migration(self, tmp_path: Path) -> None:
        """Test running migration manually."""
        tiered = TieredStorage(
            [
                TierConfig(
                    StorageTier.HOT,
                    FileSystemStore(tmp_path / "hot"),
                    max_age_seconds=0,  # Immediate demotion
                ),
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
            ]
        )

        # Store item
        node = MockNode()
        await tiered.store(node, "session-1", tier=StorageTier.HOT)

        # Wait a tiny bit to ensure age > 0
        await asyncio.sleep(0.01)

        # Run migration
        result = await tiered.run_migration()

        assert isinstance(result, MigrationResult)
        assert result.duration_ms >= 0

        await tiered.close()


# =============================================================================
# Statistics Tests
# =============================================================================


class TestTieredStorageStats:
    """Tests for statistics functionality."""

    @pytest_asyncio.fixture
    async def tiered_storage(self, tmp_path: Path) -> TieredStorage:
        """Create a tiered storage for testing."""
        storage = TieredStorage(
            [
                TierConfig(StorageTier.HOT, FileSystemStore(tmp_path / "hot")),
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path / "warm")),
            ]
        )
        yield storage
        await storage.close()

    @pytest.mark.asyncio
    async def test_stats_empty(self, tiered_storage: TieredStorage) -> None:
        """Test stats with no stored items."""
        stats = await tiered_storage.stats()

        assert "hot" in stats
        assert "warm" in stats
        assert stats["hot"].total_items == 0
        assert stats["warm"].total_items == 0

    @pytest.mark.asyncio
    async def test_stats_with_items(self, tiered_storage: TieredStorage) -> None:
        """Test stats with stored items."""
        # Store in hot
        for _ in range(3):
            await tiered_storage.store(MockNode(), "session-1", tier=StorageTier.HOT)

        # Store in warm
        for _ in range(2):
            await tiered_storage.store(MockNode(), "session-1", tier=StorageTier.WARM)

        stats = await tiered_storage.stats()

        assert stats["hot"].total_items == 3
        assert stats["warm"].total_items == 2

    @pytest.mark.asyncio
    async def test_total_stats(self, tiered_storage: TieredStorage) -> None:
        """Test aggregated total stats."""
        # Store items in different tiers
        for _ in range(3):
            await tiered_storage.store(MockNode(), "session-1", tier=StorageTier.HOT)
        for _ in range(2):
            await tiered_storage.store(MockNode(), "session-1", tier=StorageTier.WARM)

        total = await tiered_storage.total_stats()

        assert total.total_items == 5
        assert total.items_by_tier["hot"] == 3
        assert total.items_by_tier["warm"] == 2


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestTieredStorageLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path) -> None:
        """Test closing tiered storage."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        )

        await tiered.start_migration_task()
        await tiered.close()

        assert tiered._closed is True
        assert tiered._migration_task is None

    @pytest.mark.asyncio
    async def test_operations_after_close_raise(self, tmp_path: Path) -> None:
        """Test that operations after close raise RuntimeError."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        )
        await tiered.close()

        node = MockNode()
        with pytest.raises(RuntimeError, match="closed"):
            await tiered.store(node, "session-1")

        with pytest.raises(RuntimeError, match="closed"):
            await tiered.retrieve(StorageKey(session_id="x", node_id=uuid4()))

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        """Test async context manager support."""
        async with TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        ) as tiered:
            node = MockNode()
            key = await tiered.store(node, "session-1")
            assert await tiered.exists(key)

        # After context exit, should be closed
        assert tiered._closed is True

    @pytest.mark.asyncio
    async def test_double_close_safe(self, tmp_path: Path) -> None:
        """Test that calling close multiple times is safe."""
        tiered = TieredStorage(
            [
                TierConfig(StorageTier.WARM, FileSystemStore(tmp_path)),
            ]
        )

        await tiered.close()
        await tiered.close()  # Should not raise

        assert tiered._closed is True


# =============================================================================
# MigrationResult Tests
# =============================================================================


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_defaults(self) -> None:
        """Test MigrationResult default values."""
        result = MigrationResult()

        assert result.items_demoted == 0
        assert result.items_promoted == 0
        assert result.items_evicted == 0
        assert result.errors == 0
        assert result.duration_ms == 0.0

    def test_with_values(self) -> None:
        """Test MigrationResult with values."""
        result = MigrationResult(
            items_demoted=5,
            items_promoted=2,
            items_evicted=1,
            errors=0,
            duration_ms=123.45,
        )

        assert result.items_demoted == 5
        assert result.items_promoted == 2
        assert result.items_evicted == 1
        assert result.duration_ms == 123.45
