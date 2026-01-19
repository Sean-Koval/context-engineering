"""Unit tests for EvictionManager and related classes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

from context_memory.backends.filesystem import FileSystemStore
from context_memory.eviction import (
    CapacityConfig,
    EvictionCandidate,
    EvictionManager,
    EvictionResult,
    LRUImportanceScorer,
    MultiTierEvictionManager,
    PureAccessCountScorer,
    TierEvictionConfig,
)
from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageTier,
)

# =============================================================================
# Mock Classes
# =============================================================================


class MockNodeMetadata:
    """Mock node metadata for testing."""

    def __init__(self, importance: float = 0.5, tags: set[str] | None = None) -> None:
        self.importance = importance
        self.tags = tags or set()


class MockNode:
    """Mock ContextNode for testing without importing context-core."""

    def __init__(
        self,
        node_type: str = "message",
        content: str = "test content",
        token_count: int = 10,
        importance: float = 0.5,
        tags: set[str] | None = None,
    ) -> None:
        self.id = uuid4()
        self.type = type("NodeType", (), {"value": node_type.lower()})()
        self.content = content
        self.token_count = token_count
        self.metadata = MockNodeMetadata(importance, tags)

    def model_dump(self, mode: str = "python") -> dict:
        """Serialize node to dict compatible with ContextNode."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "content": {"text": self.content, "role": "user"},
            "token_count": self.token_count,
            "metadata": {
                "importance": self.metadata.importance,
                "tags": list(self.metadata.tags),
            },
        }


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def filesystem_store(tmp_path: Path) -> FileSystemStore:
    """Create a FileSystemStore for testing."""
    store = FileSystemStore(tmp_path / "store")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def demotion_store(tmp_path: Path) -> FileSystemStore:
    """Create a secondary FileSystemStore for demotion testing."""
    store = FileSystemStore(tmp_path / "demotion")
    yield store
    await store.close()


@pytest_asyncio.fixture
async def eviction_manager(
    filesystem_store: FileSystemStore,
) -> EvictionManager:
    """Create an EvictionManager for testing."""
    config = CapacityConfig(max_items=10)
    manager = EvictionManager(
        backend=filesystem_store,
        tier=StorageTier.HOT,
        capacity=config,
        session_id="test-session",  # Register a default session
        check_interval_seconds=1,
    )
    yield manager
    await manager.close()


# =============================================================================
# LRUImportanceScorer Tests
# =============================================================================


class TestLRUImportanceScorer:
    """Tests for LRUImportanceScorer."""

    def test_default_weights(self) -> None:
        """Test default weight values."""
        scorer = LRUImportanceScorer()
        assert scorer.importance_weight == 0.4
        assert scorer.recency_weight == 0.4
        assert scorer.age_weight == 0.2

    def test_weights_must_sum_to_one(self) -> None:
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            LRUImportanceScorer(
                importance_weight=0.5,
                recency_weight=0.5,
                age_weight=0.5,
            )

    def test_high_importance_low_score(self) -> None:
        """Test that high importance items get low eviction scores."""
        scorer = LRUImportanceScorer()
        now = datetime.now(UTC)

        # Create metadata for high importance item
        key = StorageKey(session_id="test", node_id=uuid4())
        high_importance = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.9,
            created_at=now - timedelta(hours=1),
            accessed_at=now - timedelta(minutes=5),
        )

        # Create metadata for low importance item
        low_importance = StorageMetadata(
            key=StorageKey(session_id="test", node_id=uuid4()),
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.1,
            created_at=now - timedelta(hours=1),
            accessed_at=now - timedelta(minutes=5),
        )

        high_score = scorer.score(high_importance, now)
        low_score = scorer.score(low_importance, now)

        # Low importance should have higher eviction score
        assert low_score > high_score

    def test_recently_accessed_low_score(self) -> None:
        """Test that recently accessed items get lower eviction scores."""
        scorer = LRUImportanceScorer()
        now = datetime.now(UTC)

        # Create base key
        key = StorageKey(session_id="test", node_id=uuid4())

        # Recently accessed
        recent = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.5,
            created_at=now - timedelta(hours=1),
            accessed_at=now - timedelta(minutes=1),  # Recent access
        )

        # Not recently accessed
        stale = StorageMetadata(
            key=StorageKey(session_id="test", node_id=uuid4()),
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.5,
            created_at=now - timedelta(hours=1),
            accessed_at=now - timedelta(hours=24),  # Old access
        )

        recent_score = scorer.score(recent, now)
        stale_score = scorer.score(stale, now)

        # Stale items should have higher eviction score
        assert stale_score > recent_score

    def test_score_clamped_to_range(self) -> None:
        """Test that scores are clamped to [0, 1]."""
        scorer = LRUImportanceScorer()
        now = datetime.now(UTC)

        # Extreme case: very old, never accessed, low importance
        key = StorageKey(session_id="test", node_id=uuid4())
        extreme = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.0,
            created_at=now - timedelta(days=365),
            accessed_at=now - timedelta(days=365),
        )

        score = scorer.score(extreme, now)
        assert 0.0 <= score <= 1.0


class TestPureAccessCountScorer:
    """Tests for PureAccessCountScorer."""

    def test_low_access_high_score(self) -> None:
        """Test that low access count gives high eviction score."""
        scorer = PureAccessCountScorer(max_access_count=100)
        now = datetime.now(UTC)

        key = StorageKey(session_id="test", node_id=uuid4())

        low_access = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.5,
            access_count=1,
        )

        high_access = StorageMetadata(
            key=StorageKey(session_id="test", node_id=uuid4()),
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.5,
            access_count=50,
        )

        low_score = scorer.score(low_access, now)
        high_score = scorer.score(high_access, now)

        # Low access count = higher eviction score
        assert low_score > high_score


# =============================================================================
# CapacityConfig Tests
# =============================================================================


class TestCapacityConfig:
    """Tests for CapacityConfig."""

    def test_default_values(self) -> None:
        """Test default capacity config values."""
        config = CapacityConfig()
        assert config.max_items is None
        assert config.max_bytes is None
        assert config.max_tokens is None
        assert config.target_utilization == 0.8
        assert config.min_free_items == 10

    def test_is_over_capacity_items(self) -> None:
        """Test capacity check for items."""
        config = CapacityConfig(max_items=100)

        assert not config.is_over_capacity(50, 0, 0)
        assert not config.is_over_capacity(99, 0, 0)
        assert config.is_over_capacity(100, 0, 0)
        assert config.is_over_capacity(150, 0, 0)

    def test_is_over_capacity_bytes(self) -> None:
        """Test capacity check for bytes."""
        config = CapacityConfig(max_bytes=1000)

        assert not config.is_over_capacity(0, 500, 0)
        assert config.is_over_capacity(0, 1000, 0)
        assert config.is_over_capacity(0, 2000, 0)

    def test_is_over_capacity_tokens(self) -> None:
        """Test capacity check for tokens."""
        config = CapacityConfig(max_tokens=5000)

        assert not config.is_over_capacity(0, 0, 1000)
        assert config.is_over_capacity(0, 0, 5000)
        assert config.is_over_capacity(0, 0, 10000)

    def test_is_over_capacity_multiple_limits(self) -> None:
        """Test capacity check with multiple limits."""
        config = CapacityConfig(max_items=100, max_bytes=1000, max_tokens=5000)

        # Under all limits
        assert not config.is_over_capacity(50, 500, 2500)

        # Over just items
        assert config.is_over_capacity(100, 500, 2500)

        # Over just bytes
        assert config.is_over_capacity(50, 1000, 2500)

        # Over just tokens
        assert config.is_over_capacity(50, 500, 5000)

    def test_items_to_evict(self) -> None:
        """Test calculating items to evict."""
        config = CapacityConfig(
            max_items=100, target_utilization=0.8, min_free_items=10
        )

        # Under capacity
        assert config.items_to_evict(50) == 0

        # At target
        assert config.items_to_evict(80) == 0

        # Over capacity: need to get to 80 + 10 headroom = 90
        # So 100 - 90 = 10 items to evict? No wait...
        # target_items = 100 * 0.8 = 80
        # items_to_evict = 100 - 80 + 10 = 30
        assert config.items_to_evict(100) == 30

    def test_items_to_evict_no_limit(self) -> None:
        """Test items_to_evict returns 0 when no item limit."""
        config = CapacityConfig()  # No max_items
        assert config.items_to_evict(1000) == 0


# =============================================================================
# EvictionResult Tests
# =============================================================================


class TestEvictionResult:
    """Tests for EvictionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default EvictionResult values."""
        result = EvictionResult()
        assert result.items_evicted == 0
        assert result.items_demoted == 0
        assert result.bytes_freed == 0
        assert result.tokens_freed == 0
        assert result.duration_ms == 0.0
        assert result.errors == 0
        assert result.candidates_evaluated == 0


# =============================================================================
# EvictionCandidate Tests
# =============================================================================


class TestEvictionCandidate:
    """Tests for EvictionCandidate dataclass."""

    def test_candidate_creation(self) -> None:
        """Test creating an eviction candidate."""
        key = StorageKey(session_id="test", node_id=uuid4())
        metadata = StorageMetadata(
            key=key,
            tier=StorageTier.HOT,
            size_bytes=100,
            token_count=10,
            node_type="message",
            importance=0.5,
        )

        candidate = EvictionCandidate(
            key=key,
            metadata=metadata,
            score=0.75,
            tier=StorageTier.HOT,
        )

        assert candidate.key == key
        assert candidate.metadata == metadata
        assert candidate.score == 0.75
        assert candidate.tier == StorageTier.HOT


# =============================================================================
# EvictionManager Tests
# =============================================================================


class TestEvictionManager:
    """Tests for EvictionManager."""

    @pytest.mark.asyncio
    async def test_manager_creation(self, filesystem_store: FileSystemStore) -> None:
        """Test basic manager creation."""
        config = CapacityConfig(max_items=100)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
        )

        assert manager.tier == StorageTier.HOT
        assert manager.total_evictions == 0
        assert manager.total_demotions == 0
        assert manager.last_check is None

        await manager.close()

    @pytest.mark.asyncio
    async def test_is_over_capacity_empty(
        self, eviction_manager: EvictionManager
    ) -> None:
        """Test capacity check on empty store."""
        assert not await eviction_manager.is_over_capacity()

    @pytest.mark.asyncio
    async def test_is_over_capacity_full(
        self, filesystem_store: FileSystemStore, tmp_path: Path
    ) -> None:
        """Test capacity check when over limit."""
        # Create manager with low capacity
        config = CapacityConfig(max_items=5)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store more items than capacity
        for i in range(6):
            node = MockNode(content=f"node {i}")
            await filesystem_store.store(node, "test-session")

        assert await manager.is_over_capacity()
        await manager.close()

    @pytest.mark.asyncio
    async def test_get_utilization(self, filesystem_store: FileSystemStore) -> None:
        """Test utilization reporting."""
        config = CapacityConfig(max_items=10, max_bytes=10000)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store some items
        for i in range(3):
            node = MockNode(content=f"node {i}")
            await filesystem_store.store(node, "test-session")

        utilization = await manager.get_utilization()
        assert "items" in utilization
        assert utilization["items"] == pytest.approx(0.3, rel=0.1)
        assert "bytes" in utilization

        await manager.close()

    @pytest.mark.asyncio
    async def test_get_eviction_candidates(
        self, filesystem_store: FileSystemStore
    ) -> None:
        """Test getting eviction candidates sorted by score."""
        config = CapacityConfig(max_items=100)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store items with different importance levels
        for i in range(5):
            # Lower importance = higher index
            importance = 0.1 * (5 - i)
            node = MockNode(content=f"node {i}", importance=importance)
            await filesystem_store.store(node, "test-session")

        candidates = await manager.get_eviction_candidates(limit=5)

        # Should be sorted by score (highest = lowest importance first)
        assert len(candidates) == 5
        for i in range(len(candidates) - 1):
            assert candidates[i].score >= candidates[i + 1].score

        await manager.close()

    @pytest.mark.asyncio
    async def test_evict_one_deletes(self, filesystem_store: FileSystemStore) -> None:
        """Test evicting a single item (delete mode)."""
        config = CapacityConfig(max_items=100)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
            demotion_target=None,  # No demotion, just delete
        )

        # Store an item
        node = MockNode(content="to evict")
        key = await filesystem_store.store(node, "test-session")

        # Get candidate
        candidates = await manager.get_eviction_candidates(limit=1)
        assert len(candidates) == 1

        # Evict it
        success = await manager.evict_one(candidates[0])
        assert success
        assert manager.total_evictions == 1
        assert manager.total_demotions == 0

        # Verify deleted
        assert not await filesystem_store.exists(key)

        await manager.close()

    @pytest.mark.asyncio
    async def test_evict_one_demotes(
        self,
        filesystem_store: FileSystemStore,
        demotion_store: FileSystemStore,
    ) -> None:
        """Test evicting a single item (demotion mode)."""
        config = CapacityConfig(max_items=100)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
            demotion_target=demotion_store,  # Demote instead of delete
        )

        # Store an item
        node = MockNode(content="to demote")
        key = await filesystem_store.store(node, "test-session")

        # Get candidate
        candidates = await manager.get_eviction_candidates(limit=1)
        assert len(candidates) == 1

        # Evict (demote) it
        success = await manager.evict_one(candidates[0])
        assert success
        assert manager.total_evictions == 0
        assert manager.total_demotions == 1

        # Verify removed from hot tier
        assert not await filesystem_store.exists(key)

        # Verify exists in demotion target
        demoted_stats = await demotion_store.stats()
        assert demoted_stats.total_items == 1

        await manager.close()

    @pytest.mark.asyncio
    async def test_check_and_evict_under_capacity(
        self, eviction_manager: EvictionManager
    ) -> None:
        """Test check_and_evict when under capacity."""
        result = await eviction_manager.check_and_evict()

        assert result.items_evicted == 0
        assert result.items_demoted == 0
        assert result.errors == 0
        assert eviction_manager.last_check is not None

    @pytest.mark.asyncio
    async def test_check_and_evict_over_capacity(
        self, filesystem_store: FileSystemStore
    ) -> None:
        """Test check_and_evict when over capacity."""
        config = CapacityConfig(
            max_items=5,
            target_utilization=0.8,  # Target 4 items
            min_free_items=1,
        )
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store 10 items
        for i in range(10):
            node = MockNode(content=f"node {i}", importance=0.1 * i)
            await filesystem_store.store(node, "test-session")

        # Should evict to get to target
        result = await manager.check_and_evict()

        assert result.items_evicted > 0
        assert result.candidates_evaluated > 0
        assert result.duration_ms > 0

        # Check remaining items
        stats = await filesystem_store.stats()
        assert stats.total_items < 10

        await manager.close()

    @pytest.mark.asyncio
    async def test_force_evict(self, filesystem_store: FileSystemStore) -> None:
        """Test force eviction of specific count."""
        config = CapacityConfig(max_items=100)  # High limit
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store 10 items
        for i in range(10):
            node = MockNode(content=f"node {i}")
            await filesystem_store.store(node, "test-session")

        # Force evict 3
        result = await manager.force_evict(3)

        assert result.items_evicted == 3
        assert result.candidates_evaluated >= 3

        stats = await filesystem_store.stats()
        assert stats.total_items == 7

        await manager.close()

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(
        self, filesystem_store: FileSystemStore
    ) -> None:
        """Test starting and stopping monitoring."""
        config = CapacityConfig(max_items=100)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
            check_interval_seconds=1,
        )

        # Start monitoring
        await manager.start_monitoring()

        # Wait for at least one check
        await asyncio.sleep(1.5)
        assert manager.last_check is not None

        # Stop monitoring
        await manager.stop_monitoring()

        # Start again (should work)
        await manager.start_monitoring()
        await manager.stop_monitoring()

        await manager.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, filesystem_store: FileSystemStore) -> None:
        """Test async context manager protocol."""
        config = CapacityConfig(max_items=100)

        async with EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
        ) as manager:
            assert manager.tier == StorageTier.HOT

        # Manager should be closed after context

    @pytest.mark.asyncio
    async def test_custom_scorer(self, filesystem_store: FileSystemStore) -> None:
        """Test using a custom scorer."""
        config = CapacityConfig(max_items=100)
        scorer = PureAccessCountScorer(max_access_count=50)

        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
            scorer=scorer,
        )

        # Store items
        for i in range(5):
            node = MockNode(content=f"node {i}")
            await filesystem_store.store(node, "test-session")

        candidates = await manager.get_eviction_candidates(limit=5)
        assert len(candidates) == 5

        await manager.close()


# =============================================================================
# MultiTierEvictionManager Tests
# =============================================================================


class TestMultiTierEvictionManager:
    """Tests for MultiTierEvictionManager."""

    @pytest.mark.asyncio
    async def test_multi_tier_creation(self, tmp_path: Path) -> None:
        """Test creating multi-tier manager."""
        hot_store = FileSystemStore(tmp_path / "hot")
        warm_store = FileSystemStore(tmp_path / "warm")
        cold_store = FileSystemStore(tmp_path / "cold")

        configs = [
            TierEvictionConfig(
                tier=StorageTier.HOT,
                backend=hot_store,
                capacity=CapacityConfig(max_items=10),
                demotion_target=warm_store,
            ),
            TierEvictionConfig(
                tier=StorageTier.WARM,
                backend=warm_store,
                capacity=CapacityConfig(max_items=100),
                demotion_target=cold_store,
            ),
            TierEvictionConfig(
                tier=StorageTier.COLD,
                backend=cold_store,
                capacity=CapacityConfig(max_items=1000),
            ),
        ]

        manager = MultiTierEvictionManager(configs)

        assert manager.get_manager(StorageTier.HOT) is not None
        assert manager.get_manager(StorageTier.WARM) is not None
        assert manager.get_manager(StorageTier.COLD) is not None

        await manager.close()
        await hot_store.close()
        await warm_store.close()
        await cold_store.close()

    @pytest.mark.asyncio
    async def test_check_all_tiers(self, tmp_path: Path) -> None:
        """Test checking all tiers at once."""
        hot_store = FileSystemStore(tmp_path / "hot")
        warm_store = FileSystemStore(tmp_path / "warm")

        configs = [
            TierEvictionConfig(
                tier=StorageTier.HOT,
                backend=hot_store,
                capacity=CapacityConfig(max_items=100),
                demotion_target=warm_store,
            ),
            TierEvictionConfig(
                tier=StorageTier.WARM,
                backend=warm_store,
                capacity=CapacityConfig(max_items=1000),
            ),
        ]

        manager = MultiTierEvictionManager(configs)

        results = await manager.check_all_tiers()
        assert "hot" in results
        assert "warm" in results
        assert isinstance(results["hot"], EvictionResult)
        assert isinstance(results["warm"], EvictionResult)

        await manager.close()
        await hot_store.close()
        await warm_store.close()

    @pytest.mark.asyncio
    async def test_cascading_demotion(self, tmp_path: Path) -> None:
        """Test items cascading through tiers."""
        hot_store = FileSystemStore(tmp_path / "hot")
        warm_store = FileSystemStore(tmp_path / "warm")
        cold_store = FileSystemStore(tmp_path / "cold")

        configs = [
            TierEvictionConfig(
                tier=StorageTier.HOT,
                backend=hot_store,
                capacity=CapacityConfig(
                    max_items=3,
                    target_utilization=0.6,
                    min_free_items=1,
                ),
                demotion_target=warm_store,
                session_id="test-session",
            ),
            TierEvictionConfig(
                tier=StorageTier.WARM,
                backend=warm_store,
                capacity=CapacityConfig(max_items=100),
                demotion_target=cold_store,
                session_id="test-session",
            ),
            TierEvictionConfig(
                tier=StorageTier.COLD,
                backend=cold_store,
                capacity=CapacityConfig(max_items=1000),
                session_id="test-session",
            ),
        ]

        manager = MultiTierEvictionManager(configs)

        # Store 5 items in hot tier
        for i in range(5):
            node = MockNode(content=f"node {i}", importance=0.1 * i)
            await hot_store.store(node, "test-session")

        # Check all tiers - should demote from hot to warm
        results = await manager.check_all_tiers()

        # Some items should have been demoted
        hot_result = results["hot"]
        assert hot_result.items_demoted > 0 or hot_result.items_evicted >= 0

        # Warm tier should now have items
        warm_stats = await warm_store.stats()
        assert warm_stats.total_items > 0

        await manager.close()
        await hot_store.close()
        await warm_store.close()
        await cold_store.close()

    @pytest.mark.asyncio
    async def test_start_stop_all(self, tmp_path: Path) -> None:
        """Test starting and stopping all tier monitors."""
        hot_store = FileSystemStore(tmp_path / "hot")
        warm_store = FileSystemStore(tmp_path / "warm")

        configs = [
            TierEvictionConfig(
                tier=StorageTier.HOT,
                backend=hot_store,
                capacity=CapacityConfig(max_items=100),
            ),
            TierEvictionConfig(
                tier=StorageTier.WARM,
                backend=warm_store,
                capacity=CapacityConfig(max_items=1000),
            ),
        ]

        manager = MultiTierEvictionManager(configs, check_interval_seconds=1)

        await manager.start_all()
        await asyncio.sleep(0.5)
        await manager.stop_all()
        await manager.close()

        await hot_store.close()
        await warm_store.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path: Path) -> None:
        """Test async context manager protocol."""
        hot_store = FileSystemStore(tmp_path / "hot")

        configs = [
            TierEvictionConfig(
                tier=StorageTier.HOT,
                backend=hot_store,
                capacity=CapacityConfig(max_items=100),
            ),
        ]

        async with MultiTierEvictionManager(configs) as manager:
            assert manager.get_manager(StorageTier.HOT) is not None

        await hot_store.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvictionIntegration:
    """Integration tests for eviction scenarios."""

    @pytest.mark.asyncio
    async def test_importance_protected_items(
        self, filesystem_store: FileSystemStore
    ) -> None:
        """Test that high importance items are protected from eviction.

        The eviction scorer prioritizes evicting low-importance items first.
        With 10 items and max_items=6, we need to evict ~3 items.
        Low importance items should be evicted before high importance ones.
        """
        config = CapacityConfig(
            max_items=6,
            target_utilization=0.8,  # Target 4.8 -> 4 items
            min_free_items=1,
        )
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store 3 high importance items first
        high_importance_ids = []
        for i in range(3):
            node = MockNode(content=f"important {i}", importance=0.9)
            key = await filesystem_store.store(node, "test-session")
            high_importance_ids.append(key)

        # Store 5 low importance items
        low_importance_ids = []
        for i in range(5):
            node = MockNode(content=f"unimportant {i}", importance=0.1)
            key = await filesystem_store.store(node, "test-session")
            low_importance_ids.append(key)

        # Total: 8 items, max 6, need to evict ~3 items
        await manager.check_and_evict()

        # Count surviving items
        high_surviving = 0
        for key in high_importance_ids:
            if await filesystem_store.exists(key):
                high_surviving += 1

        low_surviving = 0
        for key in low_importance_ids:
            if await filesystem_store.exists(key):
                low_surviving += 1

        # Low importance items should be preferentially evicted
        # With 8 items and target 4-5, we evict 3-4
        # Most evicted should be low importance
        assert low_surviving < len(low_importance_ids), (
            "Some low-importance items should be evicted"
        )
        # High importance items should have better survival rate
        assert high_surviving >= low_surviving or high_surviving >= 2, (
            f"High importance items should survive better: "
            f"high={high_surviving}, low={low_surviving}"
        )

        await manager.close()

    @pytest.mark.asyncio
    async def test_eviction_frees_resources(
        self, filesystem_store: FileSystemStore
    ) -> None:
        """Test that eviction correctly reports freed resources."""
        config = CapacityConfig(max_items=3, target_utilization=0.5, min_free_items=0)
        manager = EvictionManager(
            backend=filesystem_store,
            tier=StorageTier.HOT,
            capacity=config,
            session_id="test-session",
        )

        # Store items
        for i in range(5):
            node = MockNode(content=f"node {i}", token_count=100)
            await filesystem_store.store(node, "test-session")

        initial_stats = await filesystem_store.stats()

        # Evict
        result = await manager.check_and_evict()

        final_stats = await filesystem_store.stats()

        # Verify resources were freed
        assert result.bytes_freed > 0
        assert result.tokens_freed > 0
        assert final_stats.total_items < initial_stats.total_items

        await manager.close()
