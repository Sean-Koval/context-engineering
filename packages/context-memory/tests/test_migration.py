"""Unit tests for MigrationManager."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from context_memory.backends.filesystem import FileSystemStore
from context_memory.migration import (
    MigrationManager,
    MigrationPolicy,
    MigrationStats,
    TierMigrationConfig,
    TierMigrationCoordinator,
)

# =============================================================================
# Mock Classes
# =============================================================================


class MockNodeMetadata:
    """Mock node metadata."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNode:
    """Mock ContextNode for testing."""

    def __init__(
        self,
        content: str = "test",
        token_count: int = 10,
        importance: float = 0.5,
    ) -> None:
        self.id = uuid4()
        self.type = type("NodeType", (), {"value": "message"})()
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance)

    def model_dump(self, mode: str = "python") -> dict:
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": {"text": self.content, "role": "user"},
            "token_count": self.token_count,
            "metadata": {
                "importance": self.metadata.importance,
                "tags": [],
            },
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def source_store(tmp_path: Path) -> FileSystemStore:
    """Source store for migration."""
    store = FileSystemStore(tmp_path / "source")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def target_store(tmp_path: Path) -> FileSystemStore:
    """Target store for migration."""
    store = FileSystemStore(tmp_path / "target")
    yield store
    await store.close()


# =============================================================================
# MigrationPolicy Tests
# =============================================================================


class TestMigrationPolicy:
    """Tests for MigrationPolicy."""

    def test_default_values(self) -> None:
        """Test default policy values."""
        policy = MigrationPolicy()
        assert policy.promote_access_count == 5
        assert policy.promote_recency_hours == 1.0
        assert policy.demote_age_hours == 24.0
        assert policy.demote_min_importance == 0.3
        assert policy.batch_size == 100

    def test_custom_values(self) -> None:
        """Test custom policy values."""
        policy = MigrationPolicy(
            promote_access_count=10,
            demote_age_hours=48.0,
            batch_size=50,
        )
        assert policy.promote_access_count == 10
        assert policy.demote_age_hours == 48.0
        assert policy.batch_size == 50


# =============================================================================
# MigrationStats Tests
# =============================================================================


class TestMigrationStats:
    """Tests for MigrationStats."""

    def test_default_values(self) -> None:
        """Test default stats values."""
        stats = MigrationStats()
        assert stats.promoted == 0
        assert stats.demoted == 0
        assert stats.skipped == 0
        assert stats.errors == 0
        assert stats.duration_ms == 0.0


# =============================================================================
# MigrationManager Tests
# =============================================================================


class TestMigrationManager:
    """Tests for MigrationManager."""

    @pytest.mark.asyncio
    async def test_creation(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test manager creation."""
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
            session_id="test-session",
        )
        assert manager.total_promoted == 0
        assert manager.total_demoted == 0
        assert manager.last_run is None
        await manager.close()

    @pytest.mark.asyncio
    async def test_invalid_direction(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be"):
            MigrationManager(
                source=source_store,
                target=target_store,
                direction="invalid",
            )

    @pytest.mark.asyncio
    async def test_session_registration(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test session registration."""
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
        )
        manager.register_session("session-1")
        manager.register_session("session-2")
        manager.unregister_session("session-1")

        # Run should work with registered sessions
        stats = await manager.run()
        assert stats.errors == 0

        await manager.close()

    @pytest.mark.asyncio
    async def test_run_empty(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test run with no sessions returns quickly."""
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
        )
        stats = await manager.run()
        assert stats.promoted == 0
        assert stats.demoted == 0
        assert stats.duration_ms > 0
        await manager.close()

    @pytest.mark.asyncio
    async def test_promote_high_access_items(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test promotion of frequently accessed items."""
        policy = MigrationPolicy(promote_access_count=3, promote_recency_hours=24.0)
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
            policy=policy,
            session_id="test-session",
        )

        # Store items with different access counts
        node1 = MockNode(content="low-access")
        await source_store.store(node1, "test-session")  # Low-access item stays

        node2 = MockNode(content="high-access")
        key2 = await source_store.store(node2, "test-session")

        # Simulate high access count by updating metadata
        metadata2 = await source_store.get_metadata(key2)
        if metadata2:
            for _ in range(5):
                metadata2.touch()
            await source_store.update_metadata(
                key2,
                {
                    "access_count": metadata2.access_count,
                    "accessed_at": metadata2.accessed_at.isoformat(),
                },
            )

        stats = await manager.run()

        # High-access item should be promoted
        assert stats.promoted == 1
        assert stats.skipped == 1

        # Verify item moved
        assert not await source_store.exists(key2)
        target_stats = await target_store.stats()
        assert target_stats.total_items == 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_demote_old_items(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test demotion of old items."""
        policy = MigrationPolicy(demote_age_hours=1.0)  # 1 hour threshold
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="demote",
            policy=policy,
            session_id="test-session",
        )

        # Store an item
        node = MockNode(content="old-item")
        key = await source_store.store(node, "test-session")

        # Make item appear old by backdating created_at
        metadata = await source_store.get_metadata(key)
        if metadata:
            old_time = datetime.now(UTC) - timedelta(hours=2)  # 2 hours old
            await source_store.update_metadata(
                key, {"created_at": old_time.isoformat()}
            )

        stats = await manager.run()
        assert stats.demoted == 1

        # Item should be in target
        assert not await source_store.exists(key)
        target_stats = await target_store.stats()
        assert target_stats.total_items == 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_demote_low_importance(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test demotion of low importance items."""
        policy = MigrationPolicy(
            demote_age_hours=1000,  # High age threshold
            demote_min_importance=0.5,  # Items below 0.5 demote faster
        )
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="demote",
            policy=policy,
            session_id="test-session",
        )

        # Store low and high importance items
        low_node = MockNode(content="low-importance", importance=0.1)
        low_key = await source_store.store(low_node, "test-session")

        high_node = MockNode(content="high-importance", importance=0.9)
        await source_store.store(high_node, "test-session")

        # Update low importance item to look older (accessed long ago)
        metadata = await source_store.get_metadata(low_key)
        if metadata:
            old_time = datetime.now(UTC) - timedelta(hours=600)
            await source_store.update_metadata(
                low_key, {"accessed_at": old_time.isoformat()}
            )

        stats = await manager.run()

        # Low importance item should be demoted
        assert stats.demoted == 1
        assert stats.skipped == 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_batch_limit(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test batch size limits migrations per run."""
        policy = MigrationPolicy(demote_age_hours=0.0001, batch_size=3)
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="demote",
            policy=policy,
            session_id="test-session",
        )

        # Store 5 items
        for i in range(5):
            node = MockNode(content=f"item-{i}")
            await source_store.store(node, "test-session")

        await asyncio.sleep(0.01)
        stats = await manager.run()

        # Should only migrate batch_size items
        assert stats.demoted <= policy.batch_size

        await manager.close()

    @pytest.mark.asyncio
    async def test_background_task(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test background task starts and stops."""
        manager = MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
            session_id="test-session",
            check_interval_seconds=1,
        )

        await manager.start()
        await asyncio.sleep(0.5)
        await manager.stop()

        # Should complete without error
        await manager.close()

    @pytest.mark.asyncio
    async def test_context_manager(
        self, source_store: FileSystemStore, target_store: FileSystemStore
    ) -> None:
        """Test async context manager."""
        async with MigrationManager(
            source=source_store,
            target=target_store,
            direction="promote",
        ) as manager:
            assert manager.total_promoted == 0


# =============================================================================
# TierMigrationCoordinator Tests
# =============================================================================


class TestTierMigrationCoordinator:
    """Tests for TierMigrationCoordinator."""

    @pytest.mark.asyncio
    async def test_creation(self, tmp_path: Path) -> None:
        """Test coordinator creation."""
        store1 = FileSystemStore(tmp_path / "tier1")
        store2 = FileSystemStore(tmp_path / "tier2")

        configs = [
            TierMigrationConfig(
                source=store1,
                target=store2,
                direction="demote",
                session_id="test-session",
            ),
        ]

        coordinator = TierMigrationCoordinator(configs)
        await coordinator.close()

        await store1.close()
        await store2.close()

    @pytest.mark.asyncio
    async def test_run_all(self, tmp_path: Path) -> None:
        """Test running all migrations."""
        store1 = FileSystemStore(tmp_path / "tier1")
        store2 = FileSystemStore(tmp_path / "tier2")

        configs = [
            TierMigrationConfig(
                source=store1,
                target=store2,
                direction="demote",
                policy=MigrationPolicy(demote_age_hours=1.0),
                session_id="test-session",
            ),
        ]

        coordinator = TierMigrationCoordinator(configs)

        # Store item
        node = MockNode(content="test")
        key = await store1.store(node, "test-session")

        # Make item appear old
        old_time = datetime.now(UTC) - timedelta(hours=2)
        await store1.update_metadata(key, {"created_at": old_time.isoformat()})

        results = await coordinator.run_all()
        assert len(results) == 1
        assert results[0].demoted == 1

        await coordinator.close()
        await store1.close()
        await store2.close()

    @pytest.mark.asyncio
    async def test_session_registration(self, tmp_path: Path) -> None:
        """Test registering sessions across all managers."""
        store1 = FileSystemStore(tmp_path / "tier1")
        store2 = FileSystemStore(tmp_path / "tier2")

        configs = [
            TierMigrationConfig(store1, store2, "demote"),
        ]

        coordinator = TierMigrationCoordinator(configs)
        coordinator.register_session("new-session")
        coordinator.unregister_session("new-session")

        await coordinator.close()
        await store1.close()
        await store2.close()

    @pytest.mark.asyncio
    async def test_start_stop_all(self, tmp_path: Path) -> None:
        """Test starting and stopping all managers."""
        store1 = FileSystemStore(tmp_path / "tier1")
        store2 = FileSystemStore(tmp_path / "tier2")

        configs = [
            TierMigrationConfig(store1, store2, "demote", session_id="test"),
        ]

        coordinator = TierMigrationCoordinator(configs, check_interval_seconds=1)
        await coordinator.start_all()
        await asyncio.sleep(0.5)
        await coordinator.stop_all()
        await coordinator.close()

        await store1.close()
        await store2.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        """Test async context manager."""
        store1 = FileSystemStore(tmp_path / "tier1")
        store2 = FileSystemStore(tmp_path / "tier2")

        configs = [TierMigrationConfig(store1, store2, "demote")]

        async with TierMigrationCoordinator(configs):
            pass

        await store1.close()
        await store2.close()
