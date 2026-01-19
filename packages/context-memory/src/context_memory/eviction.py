"""EvictionManager - LRU-based eviction with importance weighting.

This module provides intelligent eviction for tiered storage, balancing
recency of access against item importance to determine eviction order.

Eviction Strategy:
    The eviction score combines LRU (Least Recently Used) with importance:

    score = (1 - importance) * age_factor + recency_factor

    Where:
    - importance: [0, 1] - higher means more valuable, less likely to evict
    - age_factor: normalized age since creation
    - recency_factor: normalized time since last access

    Higher scores = more likely to evict
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from context_memory.types import StorageKey, StorageMetadata, StorageTier

if TYPE_CHECKING:
    from context_memory.store import MemoryStore

logger = logging.getLogger(__name__)


# =============================================================================
# Eviction Score Protocol
# =============================================================================


class EvictionScorer(Protocol):
    """Protocol for custom eviction scoring strategies."""

    def score(self, metadata: StorageMetadata, now: datetime) -> float:
        """Calculate eviction score for an item.

        Args:
            metadata: Item metadata including importance, access times.
            now: Current timestamp for age calculations.

        Returns:
            Eviction score [0, 1]. Higher = more likely to evict.
        """
        ...


# =============================================================================
# Built-in Scorers
# =============================================================================


@dataclass
class LRUImportanceScorer:
    """LRU scorer weighted by importance.

    Combines recency (when last accessed) with importance to create
    an eviction score. High-importance items are protected even if
    they haven't been accessed recently.

    Attributes:
        importance_weight: Weight for importance factor [0, 1].
            Higher = importance matters more.
        recency_weight: Weight for recency factor [0, 1].
            Higher = recent access matters more.
        age_weight: Weight for absolute age factor [0, 1].
            Higher = older items more likely to evict.
        max_age_hours: Maximum age for normalization.
            Items older than this get max age score.
    """

    importance_weight: float = 0.4
    recency_weight: float = 0.4
    age_weight: float = 0.2
    max_age_hours: float = 168.0  # 1 week

    def __post_init__(self) -> None:
        """Validate weights sum to 1.0."""
        total = self.importance_weight + self.recency_weight + self.age_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def score(self, metadata: StorageMetadata, now: datetime) -> float:
        """Calculate eviction score.

        Score components:
        1. Importance factor: (1 - importance) - low importance = high score
        2. Recency factor: time since last access normalized
        3. Age factor: time since creation normalized

        Returns:
            Float [0, 1] where higher means more likely to evict.
        """
        # Importance factor: invert so low importance = high eviction score
        importance_factor = 1.0 - metadata.importance

        # Recency factor: hours since last access, normalized
        recency_hours = (now - metadata.accessed_at).total_seconds() / 3600
        recency_factor = min(recency_hours / self.max_age_hours, 1.0)

        # Age factor: hours since creation, normalized
        age_hours = (now - metadata.created_at).total_seconds() / 3600
        age_factor = min(age_hours / self.max_age_hours, 1.0)

        # Weighted combination
        score = (
            self.importance_weight * importance_factor
            + self.recency_weight * recency_factor
            + self.age_weight * age_factor
        )

        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]


@dataclass
class PureAccessCountScorer:
    """Score based purely on access count (LFU - Least Frequently Used).

    Items with fewer accesses get higher eviction scores.
    """

    max_access_count: int = 100

    def score(self, metadata: StorageMetadata, now: datetime) -> float:
        """Calculate eviction score based on access frequency.

        Returns:
            Float [0, 1] where higher = fewer accesses = more likely to evict.
        """
        normalized = min(metadata.access_count / self.max_access_count, 1.0)
        return 1.0 - normalized  # Invert: low count = high score


# =============================================================================
# Eviction Candidate
# =============================================================================


@dataclass
class EvictionCandidate:
    """An item being considered for eviction.

    Attributes:
        key: StorageKey identifying the item.
        metadata: Item metadata.
        score: Eviction score [0, 1]. Higher = more likely to evict.
        tier: Storage tier the item is in.
    """

    key: StorageKey
    metadata: StorageMetadata
    score: float
    tier: StorageTier


# =============================================================================
# Eviction Result
# =============================================================================


@dataclass
class EvictionResult:
    """Result of an eviction operation.

    Attributes:
        items_evicted: Number of items removed entirely.
        items_demoted: Number of items moved to lower tiers.
        bytes_freed: Total bytes freed from the tier.
        tokens_freed: Total tokens freed from the tier.
        duration_ms: Time taken for eviction in milliseconds.
        errors: Number of errors encountered.
        candidates_evaluated: Total candidates considered.
    """

    items_evicted: int = 0
    items_demoted: int = 0
    bytes_freed: int = 0
    tokens_freed: int = 0
    duration_ms: float = 0.0
    errors: int = 0
    candidates_evaluated: int = 0


# =============================================================================
# Tier Capacity Config
# =============================================================================


@dataclass
class CapacityConfig:
    """Capacity configuration for a storage tier.

    Defines thresholds that trigger eviction.

    Attributes:
        max_items: Maximum number of items. None = unlimited.
        max_bytes: Maximum total size in bytes. None = unlimited.
        max_tokens: Maximum total tokens. None = unlimited.
        target_utilization: Target utilization after eviction [0, 1].
            Eviction continues until usage drops below this.
        min_free_items: Minimum items to keep free for headroom.
    """

    max_items: int | None = None
    max_bytes: int | None = None
    max_tokens: int | None = None
    target_utilization: float = 0.8
    min_free_items: int = 10

    def is_over_capacity(
        self,
        current_items: int,
        current_bytes: int,
        current_tokens: int,
    ) -> bool:
        """Check if any capacity limit is exceeded.

        Returns:
            True if any limit is exceeded.
        """
        if self.max_items is not None and current_items >= self.max_items:
            return True
        if self.max_bytes is not None and current_bytes >= self.max_bytes:
            return True
        return self.max_tokens is not None and current_tokens >= self.max_tokens

    def items_to_evict(self, current_items: int) -> int:
        """Calculate how many items to evict to reach target utilization.

        Returns:
            Number of items to evict, 0 if under capacity.
        """
        if self.max_items is None:
            return 0

        target_items = int(self.max_items * self.target_utilization)
        if current_items <= target_items:
            return 0

        return current_items - target_items + self.min_free_items


# =============================================================================
# Eviction Manager
# =============================================================================


class EvictionManager:
    """Manages eviction across storage tiers.

    EvictionManager monitors tier capacity and triggers eviction when
    thresholds are exceeded. It uses a configurable scoring strategy
    to determine which items to evict first.

    Features:
        - LRU + importance weighted scoring
        - Capacity monitoring (items, bytes, tokens)
        - Tier-aware eviction (demote before evict)
        - Background monitoring task
        - Configurable eviction strategies

    Example:
        >>> from context_memory import FileSystemStore
        >>> from context_memory.eviction import EvictionManager, CapacityConfig
        >>>
        >>> store = FileSystemStore("/tmp/storage")
        >>> config = CapacityConfig(max_items=1000, max_bytes=100_000_000)
        >>>
        >>> manager = EvictionManager(
        ...     backend=store,
        ...     tier=StorageTier.HOT,
        ...     capacity=config,
        ... )
        >>>
        >>> # Check and evict if needed
        >>> result = await manager.check_and_evict()
        >>> print(f"Evicted {result.items_evicted} items")
    """

    def __init__(
        self,
        backend: MemoryStore,
        tier: StorageTier,
        capacity: CapacityConfig,
        *,
        session_id: str | None = None,
        scorer: EvictionScorer | None = None,
        demotion_target: MemoryStore | None = None,
        check_interval_seconds: int = 60,
    ) -> None:
        """Initialize EvictionManager.

        Args:
            backend: Storage backend to manage eviction for.
            tier: The tier this manager is responsible for.
            capacity: Capacity thresholds and target utilization.
            session_id: If provided, only manage eviction for this session.
                If None, tracks sessions as they are registered.
            scorer: Custom eviction scorer. Defaults to LRUImportanceScorer.
            demotion_target: Backend to demote items to instead of evicting.
                If None, items are deleted.
            check_interval_seconds: Interval for background capacity checks.
        """
        self._backend = backend
        self._tier = tier
        self._capacity = capacity
        self._scorer = scorer or LRUImportanceScorer()
        self._demotion_target = demotion_target
        self._check_interval = check_interval_seconds

        # Session tracking
        self._sessions: set[str] = set()
        if session_id:
            self._sessions.add(session_id)

        self._monitor_task: asyncio.Task[None] | None = None
        self._closed = False

        # Metrics
        self._total_evictions = 0
        self._total_demotions = 0
        self._last_check: datetime | None = None

    def register_session(self, session_id: str) -> None:
        """Register a session for eviction management.

        Args:
            session_id: The session to track.
        """
        self._sessions.add(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session from eviction management.

        Args:
            session_id: The session to stop tracking.
        """
        self._sessions.discard(session_id)

    @property
    def sessions(self) -> set[str]:
        """Set of sessions being managed."""
        return self._sessions.copy()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def tier(self) -> StorageTier:
        """The storage tier this manager handles."""
        return self._tier

    @property
    def total_evictions(self) -> int:
        """Total items evicted since creation."""
        return self._total_evictions

    @property
    def total_demotions(self) -> int:
        """Total items demoted since creation."""
        return self._total_demotions

    @property
    def last_check(self) -> datetime | None:
        """Timestamp of last capacity check."""
        return self._last_check

    # =========================================================================
    # Capacity Checking
    # =========================================================================

    async def is_over_capacity(self) -> bool:
        """Check if the tier is over capacity.

        Returns:
            True if any capacity limit is exceeded.
        """
        stats = await self._backend.stats()
        return self._capacity.is_over_capacity(
            current_items=stats.total_items,
            current_bytes=stats.total_size_bytes,
            current_tokens=stats.total_tokens,
        )

    async def get_utilization(self) -> dict[str, float]:
        """Get current utilization percentages.

        Returns:
            Dictionary with utilization for items, bytes, tokens.
            Values are [0, 1] or None if no limit configured.
        """
        stats = await self._backend.stats()
        result: dict[str, float] = {}

        if self._capacity.max_items:
            result["items"] = stats.total_items / self._capacity.max_items
        if self._capacity.max_bytes:
            result["bytes"] = stats.total_size_bytes / self._capacity.max_bytes
        if self._capacity.max_tokens:
            result["tokens"] = stats.total_tokens / self._capacity.max_tokens

        return result

    # =========================================================================
    # Eviction Operations
    # =========================================================================

    async def get_eviction_candidates(
        self,
        limit: int = 100,
        session_id: str | None = None,
    ) -> list[EvictionCandidate]:
        """Get items sorted by eviction priority.

        Retrieves metadata for items and scores them using the configured
        scorer. Returns items sorted by score (highest = evict first).

        Args:
            limit: Maximum number of candidates to return.
            session_id: If provided, only get candidates from this session.
                If None, gets candidates from all registered sessions.

        Returns:
            List of EvictionCandidate sorted by score (descending).
        """
        now = datetime.now(UTC)
        candidates: list[EvictionCandidate] = []

        # Determine which sessions to query
        sessions_to_check = [session_id] if session_id else list(self._sessions)

        if not sessions_to_check:
            return []

        # Get keys from all sessions
        # Note: We don't filter by tier here because each EvictionManager
        # is associated with a single backend that represents its tier.
        for sess_id in sessions_to_check:
            try:
                keys = await self._backend.list_keys(
                    session_id=sess_id,
                    limit=limit * 2,  # Fetch extra in case some fail
                )

                for key in keys:
                    try:
                        metadata = await self._backend.get_metadata(key)
                        if metadata is None:
                            continue

                        score = self._scorer.score(metadata, now)
                        candidates.append(
                            EvictionCandidate(
                                key=key,
                                metadata=metadata,
                                score=score,
                                tier=self._tier,
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to get metadata for {key}: {e}")
            except Exception as e:
                logger.warning(f"Failed to list keys for session {sess_id}: {e}")

        # Sort by score descending (highest = evict first)
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates[:limit]

    async def evict_one(self, candidate: EvictionCandidate) -> bool:
        """Evict or demote a single item.

        If a demotion target is configured, attempts to demote first.
        Otherwise, deletes the item.

        Args:
            candidate: The item to evict.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self._demotion_target is not None:
                # Demote to lower tier
                node = await self._backend.retrieve(candidate.key)
                if node is not None:
                    await self._demotion_target.store(
                        node, candidate.key.session_id, candidate.metadata
                    )
                    await self._backend.delete(candidate.key)
                    self._total_demotions += 1
                    logger.debug(
                        f"Demoted {candidate.key} (score={candidate.score:.3f})"
                    )
                    return True
            else:
                # Evict (delete)
                deleted = await self._backend.delete(candidate.key)
                if deleted:
                    self._total_evictions += 1
                    logger.debug(
                        f"Evicted {candidate.key} (score={candidate.score:.3f})"
                    )
                return deleted

        except Exception as e:
            logger.error(f"Failed to evict {candidate.key}: {e}")
            return False

    async def check_and_evict(self) -> EvictionResult:
        """Check capacity and evict items if over threshold.

        This is the main entry point for eviction. It:
        1. Checks current capacity utilization
        2. If over capacity, gets eviction candidates
        3. Evicts items until under target utilization

        Returns:
            EvictionResult with counts and metrics.
        """
        start = time.perf_counter()
        result = EvictionResult()
        self._last_check = datetime.now(UTC)

        try:
            stats = await self._backend.stats()

            # Check if eviction needed
            if not self._capacity.is_over_capacity(
                stats.total_items,
                stats.total_size_bytes,
                stats.total_tokens,
            ):
                result.duration_ms = (time.perf_counter() - start) * 1000
                return result

            # Calculate how many items to evict
            items_to_evict = self._capacity.items_to_evict(stats.total_items)
            if items_to_evict <= 0:
                result.duration_ms = (time.perf_counter() - start) * 1000
                return result

            logger.info(
                f"Tier {self._tier.value} over capacity: "
                f"{stats.total_items} items, need to evict {items_to_evict}"
            )

            # Get candidates
            candidates = await self.get_eviction_candidates(limit=items_to_evict * 2)
            result.candidates_evaluated = len(candidates)

            # Evict until under capacity
            evicted_count = 0
            for candidate in candidates:
                if evicted_count >= items_to_evict:
                    break

                if await self.evict_one(candidate):
                    if self._demotion_target:
                        result.items_demoted += 1
                    else:
                        result.items_evicted += 1
                    result.bytes_freed += candidate.metadata.size_bytes
                    result.tokens_freed += candidate.metadata.token_count
                    evicted_count += 1
                else:
                    result.errors += 1

        except Exception as e:
            logger.error(f"Eviction check failed: {e}")
            result.errors += 1

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    async def force_evict(self, count: int) -> EvictionResult:
        """Force eviction of a specific number of items.

        Unlike check_and_evict, this doesn't check capacity first.
        Useful for preemptive eviction or testing.

        Args:
            count: Number of items to evict.

        Returns:
            EvictionResult with counts and metrics.
        """
        start = time.perf_counter()
        result = EvictionResult()
        self._last_check = datetime.now(UTC)

        try:
            candidates = await self.get_eviction_candidates(limit=count * 2)
            result.candidates_evaluated = len(candidates)

            evicted_count = 0
            for candidate in candidates:
                if evicted_count >= count:
                    break

                if await self.evict_one(candidate):
                    if self._demotion_target:
                        result.items_demoted += 1
                    else:
                        result.items_evicted += 1
                    result.bytes_freed += candidate.metadata.size_bytes
                    result.tokens_freed += candidate.metadata.token_count
                    evicted_count += 1
                else:
                    result.errors += 1

        except Exception as e:
            logger.error(f"Force eviction failed: {e}")
            result.errors += 1

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    # =========================================================================
    # Background Monitoring
    # =========================================================================

    async def start_monitoring(self) -> None:
        """Start background capacity monitoring.

        Periodically checks capacity and triggers eviction when needed.
        """
        if self._monitor_task is not None:
            return  # Already running

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Started eviction monitor for {self._tier.value} "
            f"(interval={self._check_interval}s)"
        )

    async def stop_monitoring(self) -> None:
        """Stop background capacity monitoring."""
        if self._monitor_task is None:
            return

        self._monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._monitor_task
        self._monitor_task = None
        logger.info(f"Stopped eviction monitor for {self._tier.value}")

    async def _monitor_loop(self) -> None:
        """Background loop for capacity monitoring."""
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                result = await self.check_and_evict()
                if result.items_evicted > 0 or result.items_demoted > 0:
                    logger.info(
                        f"Eviction monitor {self._tier.value}: "
                        f"evicted={result.items_evicted}, "
                        f"demoted={result.items_demoted}, "
                        f"freed={result.bytes_freed} bytes"
                    )
            except Exception as e:
                logger.error(f"Eviction monitor error: {e}")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Stop monitoring and cleanup.

        Safe to call multiple times.
        """
        if self._closed:
            return

        await self.stop_monitoring()
        self._closed = True

    async def __aenter__(self) -> EvictionManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Multi-Tier Eviction Coordinator
# =============================================================================


@dataclass
class TierEvictionConfig:
    """Configuration for eviction within a tier.

    Attributes:
        tier: The storage tier.
        backend: Storage backend for this tier.
        capacity: Capacity limits and thresholds.
        demotion_target: Next tier to demote to (None = evict).
        session_id: If provided, only manage eviction for this session.
    """

    tier: StorageTier
    backend: Any  # MemoryStore
    capacity: CapacityConfig
    demotion_target: Any | None = None  # MemoryStore | None
    session_id: str | None = None


class MultiTierEvictionManager:
    """Coordinates eviction across multiple storage tiers.

    Creates and manages individual EvictionManagers for each tier,
    ensuring items flow properly from hot to warm to cold.

    Example:
        >>> manager = MultiTierEvictionManager([
        ...     TierEvictionConfig(
        ...         tier=StorageTier.HOT,
        ...         backend=hot_store,
        ...         capacity=CapacityConfig(max_items=100),
        ...         demotion_target=warm_store,
        ...     ),
        ...     TierEvictionConfig(
        ...         tier=StorageTier.WARM,
        ...         backend=warm_store,
        ...         capacity=CapacityConfig(max_items=1000),
        ...         demotion_target=cold_store,
        ...     ),
        ...     TierEvictionConfig(
        ...         tier=StorageTier.COLD,
        ...         backend=cold_store,
        ...         capacity=CapacityConfig(max_items=10000),
        ...     ),
        ... ])
        >>>
        >>> await manager.start_all()
        >>> # ... use storage ...
        >>> await manager.close()
    """

    def __init__(
        self,
        configs: list[TierEvictionConfig],
        *,
        scorer: EvictionScorer | None = None,
        check_interval_seconds: int = 60,
    ) -> None:
        """Initialize MultiTierEvictionManager.

        Args:
            configs: Configuration for each tier to manage.
            scorer: Shared eviction scorer for all tiers.
            check_interval_seconds: Background check interval.
        """
        self._managers: dict[StorageTier, EvictionManager] = {}

        for config in configs:
            manager = EvictionManager(
                backend=config.backend,
                tier=config.tier,
                capacity=config.capacity,
                session_id=config.session_id,
                scorer=scorer,
                demotion_target=config.demotion_target,
                check_interval_seconds=check_interval_seconds,
            )
            self._managers[config.tier] = manager

    def get_manager(self, tier: StorageTier) -> EvictionManager | None:
        """Get the manager for a specific tier."""
        return self._managers.get(tier)

    def register_session(self, session_id: str) -> None:
        """Register a session with all tier managers.

        Args:
            session_id: The session to track across all tiers.
        """
        for manager in self._managers.values():
            manager.register_session(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session from all tier managers.

        Args:
            session_id: The session to stop tracking.
        """
        for manager in self._managers.values():
            manager.unregister_session(session_id)

    async def check_all_tiers(self) -> dict[str, EvictionResult]:
        """Run eviction check on all tiers.

        Processes tiers from hot to cold to allow cascading demotion.

        Returns:
            Dictionary mapping tier name to EvictionResult.
        """
        results: dict[str, EvictionResult] = {}
        tier_order = [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]

        for tier in tier_order:
            manager = self._managers.get(tier)
            if manager:
                results[tier.value] = await manager.check_and_evict()

        return results

    async def start_all(self) -> None:
        """Start monitoring on all tiers."""
        for manager in self._managers.values():
            await manager.start_monitoring()

    async def stop_all(self) -> None:
        """Stop monitoring on all tiers."""
        for manager in self._managers.values():
            await manager.stop_monitoring()

    async def close(self) -> None:
        """Close all managers."""
        for manager in self._managers.values():
            await manager.close()

    async def __aenter__(self) -> MultiTierEvictionManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
