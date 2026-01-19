"""TieredStorage - Multi-tier storage coordinator.

This module provides a unified interface over multiple storage tiers (hot, warm, cold)
with automatic tier migration based on access patterns and age.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      TieredStorage                               │
    ├─────────────────────────────────────────────────────────────────┤
    │  HOT (Redis/Memory)    │  < 1 hour old, high access             │
    │  WARM (PostgreSQL)     │  < 24 hours, moderate access           │
    │  COLD (S3/Filesystem)  │  > 24 hours, archived                  │
    └─────────────────────────────────────────────────────────────────┘

Features:
    - Automatic promotion on access (cold → warm → hot)
    - Automatic demotion based on age/importance
    - Background migration task
    - Unified query interface across all tiers
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from context_memory.store import MemoryStore
from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Configuration for a storage tier.

    Attributes:
        tier: The storage tier this config applies to.
        backend: The MemoryStore implementation for this tier.
        max_age_seconds: Maximum age in seconds before items are demoted.
            None means no age limit.
        max_items: Maximum number of items in this tier. None means no limit.
        min_importance: Minimum importance score required to stay in this tier.
            Items below this threshold will be demoted.
        promote_on_access: Whether accessing an item promotes it to a higher tier.

    Example:
        >>> hot_tier = TierConfig(
        ...     tier=StorageTier.HOT,
        ...     backend=RedisStore("redis://localhost"),
        ...     max_age_seconds=3600,  # 1 hour
        ...     max_items=1000,
        ...     min_importance=0.3,
        ... )
    """

    tier: StorageTier
    backend: MemoryStore
    max_age_seconds: int | None = None
    max_items: int | None = None
    min_importance: float = 0.0
    promote_on_access: bool = True


@dataclass
class MigrationResult:
    """Result of a tier migration operation.

    Attributes:
        items_demoted: Number of items moved to lower tiers.
        items_promoted: Number of items moved to higher tiers.
        items_evicted: Number of items removed entirely.
        errors: Number of errors encountered during migration.
        duration_ms: Time taken for migration in milliseconds.
    """

    items_demoted: int = 0
    items_promoted: int = 0
    items_evicted: int = 0
    errors: int = 0
    duration_ms: float = 0.0


class TieredStorage:
    """Multi-tier storage with automatic migration.

    TieredStorage coordinates multiple storage backends organized by access speed
    and capacity. Items flow from hot (fast, limited) to cold (slow, unlimited)
    based on age, access patterns, and importance.

    Features:
        - Automatic promotion on access: Items retrieved from lower tiers are
          promoted to higher tiers for faster subsequent access.
        - Automatic demotion: Background task migrates items to lower tiers
          based on age and access patterns.
        - Unified interface: Single API for storing/retrieving across all tiers.
        - Configurable policies: Per-tier settings for age limits, capacity,
          and importance thresholds.

    Example:
        >>> from context_memory import RedisStore, PostgresStore, FileSystemStore
        >>>
        >>> tiered = TieredStorage([
        ...     TierConfig(StorageTier.HOT, RedisStore(...), max_age_seconds=3600),
        ...     TierConfig(StorageTier.WARM, PostgresStore(...), max_age_seconds=86400),
        ...     TierConfig(StorageTier.COLD, FileSystemStore(...)),
        ... ])
        >>>
        >>> # Start background migration
        >>> await tiered.start_migration_task()
        >>>
        >>> # Store (defaults to hot tier)
        >>> key = await tiered.store(node, "session-123")
        >>>
        >>> # Retrieve (checks all tiers, promotes if found in lower tier)
        >>> node = await tiered.retrieve(key)
        >>>
        >>> # Cleanup
        >>> await tiered.close()
    """

    # Tier order from hottest to coldest
    TIER_ORDER = [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]

    def __init__(
        self,
        tiers: list[TierConfig],
        *,
        promotion_on_access: bool = True,
        migration_interval_seconds: int = 300,
    ) -> None:
        """Initialize TieredStorage.

        Args:
            tiers: List of TierConfig objects, one per tier to use.
                Not all tiers need to be configured.
            promotion_on_access: Global setting for whether to promote items
                on access. Can be overridden per-tier via TierConfig.
            migration_interval_seconds: Interval between background migration
                runs in seconds. Default 300 (5 minutes).

        Raises:
            ValueError: If duplicate tiers are configured.
        """
        # Validate no duplicate tiers
        tier_types = [t.tier for t in tiers]
        if len(tier_types) != len(set(tier_types)):
            raise ValueError("Duplicate tier configurations provided")

        self._tiers: dict[StorageTier, TierConfig] = {t.tier: t for t in tiers}
        self._promotion_on_access = promotion_on_access
        self._migration_interval = migration_interval_seconds
        self._migration_task: asyncio.Task[None] | None = None
        self._closed = False

    # =========================================================================
    # Tier Navigation Helpers
    # =========================================================================

    def _get_next_tier(self, tier: StorageTier) -> StorageTier | None:
        """Get the next tier for demotion (colder).

        Args:
            tier: Current tier.

        Returns:
            Next colder tier, or None if already at coldest.
        """
        try:
            idx = self.TIER_ORDER.index(tier)
            if idx < len(self.TIER_ORDER) - 1:
                next_tier = self.TIER_ORDER[idx + 1]
                # Only return if we have this tier configured
                if next_tier in self._tiers:
                    return next_tier
        except ValueError:
            pass
        return None

    def _get_prev_tier(self, tier: StorageTier) -> StorageTier | None:
        """Get the previous tier for promotion (hotter).

        Args:
            tier: Current tier.

        Returns:
            Previous hotter tier, or None if already at hottest.
        """
        try:
            idx = self.TIER_ORDER.index(tier)
            if idx > 0:
                prev_tier = self.TIER_ORDER[idx - 1]
                # Only return if we have this tier configured
                if prev_tier in self._tiers:
                    return prev_tier
        except ValueError:
            pass
        return None

    def _get_hottest_tier(self) -> StorageTier | None:
        """Get the hottest configured tier."""
        for tier in self.TIER_ORDER:
            if tier in self._tiers:
                return tier
        return None

    def _get_configured_tiers_in_order(self) -> list[StorageTier]:
        """Get configured tiers in order from hot to cold."""
        return [t for t in self.TIER_ORDER if t in self._tiers]

    # =========================================================================
    # Core Storage Operations
    # =========================================================================

    async def store(
        self,
        node: Any,  # ContextNode
        session_id: str,
        tier: StorageTier | None = None,
        metadata: StorageMetadata | None = None,
    ) -> StorageKey:
        """Store a node in the specified tier.

        Args:
            node: The ContextNode to store.
            session_id: Session identifier for namespacing.
            tier: Target tier. Defaults to hottest configured tier.
            metadata: Optional metadata. Will be auto-generated if not provided.

        Returns:
            StorageKey for retrieval.

        Raises:
            ValueError: If specified tier is not configured.
            RuntimeError: If no tiers are configured.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        # Default to hottest tier
        if tier is None:
            tier = self._get_hottest_tier()
            if tier is None:
                raise RuntimeError("No tiers configured")

        config = self._tiers.get(tier)
        if not config:
            raise ValueError(f"Tier {tier} not configured")

        # Update metadata tier if provided
        if metadata is not None:
            # Create a copy with updated tier
            metadata_dict = metadata.model_dump()
            metadata_dict["tier"] = tier
            metadata = StorageMetadata.model_validate(metadata_dict)

        return await config.backend.store(node, session_id, metadata)

    async def store_batch(
        self,
        nodes: list[Any],  # list[ContextNode]
        session_id: str,
        tier: StorageTier | None = None,
    ) -> list[StorageKey]:
        """Store multiple nodes in the specified tier.

        Args:
            nodes: List of ContextNodes to store.
            session_id: Session identifier.
            tier: Target tier. Defaults to hottest configured tier.

        Returns:
            List of StorageKeys in same order as input.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        if tier is None:
            tier = self._get_hottest_tier()
            if tier is None:
                raise RuntimeError("No tiers configured")

        config = self._tiers.get(tier)
        if not config:
            raise ValueError(f"Tier {tier} not configured")

        return await config.backend.store_batch(nodes, session_id)

    async def retrieve(
        self,
        key: StorageKey,
        *,
        promote: bool | None = None,
    ) -> Any | None:  # ContextNode | None
        """Retrieve a node, checking all tiers.

        Searches tiers from hottest to coldest. If found in a lower tier
        and promotion is enabled, the item is promoted to the hottest tier.

        Args:
            key: StorageKey to retrieve.
            promote: Whether to promote to hottest tier on access.
                Defaults to the global promotion_on_access setting.

        Returns:
            The ContextNode if found, None otherwise.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        should_promote = promote if promote is not None else self._promotion_on_access

        # Search tiers in order from hot to cold
        for tier in self._get_configured_tiers_in_order():
            config = self._tiers[tier]
            node = await config.backend.retrieve(key)

            if node is not None:
                # Promote if needed and not already in hottest tier
                hottest = self._get_hottest_tier()
                if should_promote and tier != hottest and hottest is not None:
                    await self._promote_item(key, node, tier, hottest)

                return node

        return None

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
        *,
        promote: bool | None = None,
    ) -> list[Any | None]:  # list[ContextNode | None]
        """Retrieve multiple nodes.

        Args:
            keys: List of StorageKeys to retrieve.
            promote: Whether to promote on access.

        Returns:
            List of ContextNodes (or None) in same order as input keys.
        """
        # For simplicity, use sequential retrieves
        # Could be optimized with parallel tier queries
        return [await self.retrieve(key, promote=promote) for key in keys]

    async def delete(self, key: StorageKey) -> bool:
        """Delete a node from all tiers.

        Args:
            key: StorageKey to delete.

        Returns:
            True if the node existed in any tier and was deleted.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        deleted = False
        for config in self._tiers.values():
            if await config.backend.delete(key):
                deleted = True

        return deleted

    async def exists(self, key: StorageKey) -> bool:
        """Check if a key exists in any tier.

        Args:
            key: StorageKey to check.

        Returns:
            True if the key exists in any tier.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        for config in self._tiers.values():
            if await config.backend.exists(key):
                return True
        return False

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(self, key: StorageKey) -> StorageMetadata | None:
        """Get metadata from the first tier containing the key.

        Args:
            key: StorageKey to get metadata for.

        Returns:
            StorageMetadata if found, None otherwise.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        for config in self._tiers.values():
            metadata = await config.backend.get_metadata(key)
            if metadata is not None:
                return metadata
        return None

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict[str, Any],
    ) -> bool:
        """Update metadata in the tier containing the key.

        Args:
            key: StorageKey to update.
            updates: Dictionary of field updates.

        Returns:
            True if found and updated, False otherwise.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        for config in self._tiers.values():
            if await config.backend.exists(key):
                return await config.backend.update_metadata(key, updates)
        return False

    # =========================================================================
    # Tier Migration
    # =========================================================================

    async def _promote_item(
        self,
        key: StorageKey,
        node: Any,  # ContextNode
        from_tier: StorageTier,
        to_tier: StorageTier,
    ) -> bool:
        """Promote an item to a higher (hotter) tier.

        Args:
            key: StorageKey of the item.
            node: The ContextNode to promote.
            from_tier: Current tier.
            to_tier: Target tier (must be hotter).

        Returns:
            True if promotion succeeded.
        """
        to_config = self._tiers.get(to_tier)
        from_config = self._tiers.get(from_tier)

        if not to_config or not from_config:
            return False

        try:
            # Store in target tier
            await to_config.backend.store(node, key.session_id)
            # Delete from source tier
            await from_config.backend.delete(key)
            logger.debug(f"Promoted {key} from {from_tier} to {to_tier}")
            return True
        except Exception as e:
            logger.warning(f"Failed to promote {key}: {e}")
            return False

    async def migrate_to_tier(
        self,
        key: StorageKey,
        target_tier: StorageTier,
    ) -> bool:
        """Manually migrate an item to a specific tier.

        Args:
            key: StorageKey of the item to migrate.
            target_tier: Target tier.

        Returns:
            True if migration succeeded, False if item not found or failed.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        if target_tier not in self._tiers:
            raise ValueError(f"Tier {target_tier} not configured")

        # Find current tier
        current_tier = None
        node = None

        for tier in self._get_configured_tiers_in_order():
            config = self._tiers[tier]
            node = await config.backend.retrieve(key)
            if node is not None:
                current_tier = tier
                break

        if node is None or current_tier is None:
            return False

        if current_tier == target_tier:
            return True  # Already in target tier

        # Store in target tier
        target_config = self._tiers[target_tier]
        current_config = self._tiers[current_tier]

        try:
            await target_config.backend.store(node, key.session_id)
            await current_config.backend.delete(key)
            logger.info(f"Migrated {key} from {current_tier} to {target_tier}")
            return True
        except Exception as e:
            logger.error(f"Failed to migrate {key}: {e}")
            return False

    # =========================================================================
    # Background Migration Task
    # =========================================================================

    async def start_migration_task(self) -> None:
        """Start the background tier migration task.

        The migration task runs periodically to:
        - Demote old items from hot to warm to cold
        - Evict items that exceed tier capacity
        """
        if self._migration_task is not None:
            return  # Already running

        self._migration_task = asyncio.create_task(self._migration_loop())
        logger.info(f"Started migration task with interval {self._migration_interval}s")

    async def stop_migration_task(self) -> None:
        """Stop the background migration task."""
        if self._migration_task is None:
            return

        self._migration_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._migration_task
        self._migration_task = None
        logger.info("Stopped migration task")

    async def _migration_loop(self) -> None:
        """Background loop for tier migration."""
        while True:
            await asyncio.sleep(self._migration_interval)
            try:
                result = await self.run_migration()
                if result.items_demoted > 0 or result.items_evicted > 0:
                    logger.info(
                        f"Migration complete: demoted={result.items_demoted}, "
                        f"evicted={result.items_evicted}, errors={result.errors}"
                    )
            except Exception as e:
                logger.error(f"Migration error: {e}")

    async def run_migration(self) -> MigrationResult:
        """Run a single migration pass.

        Checks each tier for items that should be demoted based on:
        - Age exceeding max_age_seconds
        - Importance below min_importance threshold

        Returns:
            MigrationResult with counts of items moved/evicted.
        """
        import time

        start = time.perf_counter()
        result = MigrationResult()
        now = datetime.now(UTC)

        # Process tiers from hot to warm (skip cold, nothing to demote from)
        for tier in [StorageTier.HOT, StorageTier.WARM]:
            config = self._tiers.get(tier)
            if not config:
                continue

            next_tier = self._get_next_tier(tier)
            next_config = self._tiers.get(next_tier) if next_tier else None

            # Get all items in this tier
            # For production, this should be paginated or use a streaming approach
            try:
                stats = await config.backend.stats()
                if stats.total_items == 0:
                    continue

                # Get keys with metadata for filtering
                # Note: In a real implementation, you'd want a more efficient
                # way to query items eligible for demotion
                keys = await config.backend.list_keys(
                    session_id="",  # Empty string gets all sessions for some backends
                    tier=tier,
                    limit=1000,
                )

                for key in keys:
                    metadata = await config.backend.get_metadata(key)
                    if not metadata:
                        continue

                    should_demote = False
                    reason = ""

                    # Check age
                    if config.max_age_seconds:
                        age = (now - metadata.created_at).total_seconds()
                        if age > config.max_age_seconds:
                            should_demote = True
                            reason = f"age {age:.0f}s > {config.max_age_seconds}s"

                    # Check importance threshold
                    if (
                        not should_demote
                        and metadata.importance < config.min_importance
                    ):
                        should_demote = True
                        reason = (
                            f"importance {metadata.importance} "
                            f"< {config.min_importance}"
                        )

                    if should_demote:
                        if next_config:
                            # Demote to next tier
                            node = await config.backend.retrieve(key)
                            if node:
                                try:
                                    await next_config.backend.store(
                                        node, key.session_id
                                    )
                                    await config.backend.delete(key)
                                    result.items_demoted += 1
                                    logger.debug(
                                        "Demoted %s from %s to %s: %s",
                                        key,
                                        tier,
                                        next_tier,
                                        reason,
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to demote {key}: {e}")
                                    result.errors += 1
                        else:
                            # No lower tier, evict
                            try:
                                await config.backend.delete(key)
                                result.items_evicted += 1
                                logger.debug(f"Evicted {key} from {tier}: {reason}")
                            except Exception as e:
                                logger.warning(f"Failed to evict {key}: {e}")
                                result.errors += 1

            except Exception as e:
                logger.error(f"Error processing tier {tier}: {e}")
                result.errors += 1

        result.duration_ms = (time.perf_counter() - start) * 1000
        return result

    # =========================================================================
    # Statistics & Lifecycle
    # =========================================================================

    async def stats(self, session_id: str | None = None) -> dict[str, StorageStats]:
        """Get statistics for all tiers.

        Args:
            session_id: If provided, get stats for this session only.

        Returns:
            Dictionary mapping tier name to StorageStats.
        """
        if self._closed:
            raise RuntimeError("TieredStorage is closed")

        result = {}
        for tier, config in self._tiers.items():
            result[tier.value] = await config.backend.stats(session_id)
        return result

    async def total_stats(self, session_id: str | None = None) -> StorageStats:
        """Get aggregated statistics across all tiers.

        Args:
            session_id: If provided, get stats for this session only.

        Returns:
            Aggregated StorageStats.
        """
        tier_stats = await self.stats(session_id)

        total_items = 0
        total_size = 0
        total_tokens = 0
        items_by_tier: dict[str, int] = {}
        size_by_tier: dict[str, int] = {}
        access_counts: list[float] = []
        oldest: datetime | None = None
        newest: datetime | None = None

        for tier_name, stats in tier_stats.items():
            total_items += stats.total_items
            total_size += stats.total_size_bytes
            total_tokens += stats.total_tokens
            items_by_tier[tier_name] = stats.total_items
            size_by_tier[tier_name] = stats.total_size_bytes

            if stats.avg_access_count > 0:
                access_counts.append(stats.avg_access_count)

            if stats.oldest_item and (oldest is None or stats.oldest_item < oldest):
                oldest = stats.oldest_item
            if stats.newest_item and (newest is None or stats.newest_item > newest):
                newest = stats.newest_item

        return StorageStats(
            total_items=total_items,
            total_size_bytes=total_size,
            total_tokens=total_tokens,
            items_by_tier=items_by_tier,
            size_by_tier=size_by_tier,
            avg_access_count=sum(access_counts) / len(access_counts)
            if access_counts
            else 0.0,
            oldest_item=oldest,
            newest_item=newest,
        )

    async def close(self) -> None:
        """Close all backends and stop background tasks.

        Safe to call multiple times.
        """
        if self._closed:
            return

        await self.stop_migration_task()

        for config in self._tiers.values():
            try:
                await config.backend.close()
            except Exception as e:
                logger.warning(f"Error closing {config.tier} backend: {e}")

        self._closed = True
        logger.info("TieredStorage closed")

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> TieredStorage:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
