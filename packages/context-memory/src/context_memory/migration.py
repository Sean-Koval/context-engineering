"""MigrationManager - Policy-based tier migration.

Provides proactive promotion based on access patterns and configurable
demotion policies. Works alongside TieredStorage to optimize data placement.

Key differences from TieredStorage.run_migration():
- Proactive promotion (not just on-access)
- Access-count-based decisions
- Configurable policy thresholds
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from context_memory.types import StorageKey

if TYPE_CHECKING:
    from context_memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class MigrationPolicy:
    """Policy for tier migration decisions.

    Attributes:
        promote_access_count: Minimum access count to trigger promotion.
        promote_recency_hours: Item must be accessed within this window to promote.
        demote_age_hours: Hours before item is eligible for demotion.
        demote_min_importance: Items below this importance demote faster.
        batch_size: Max items to migrate per run.
    """

    promote_access_count: int = 5
    promote_recency_hours: float = 1.0
    demote_age_hours: float = 24.0
    demote_min_importance: float = 0.3
    batch_size: int = 100


@dataclass
class MigrationStats:
    """Statistics from a migration run.

    Attributes:
        promoted: Items moved to higher tiers.
        demoted: Items moved to lower tiers.
        skipped: Items that didn't meet criteria.
        errors: Migration failures.
        duration_ms: Time taken in milliseconds.
    """

    promoted: int = 0
    demoted: int = 0
    skipped: int = 0
    errors: int = 0
    duration_ms: float = 0.0


class MigrationManager:
    """Manages policy-based tier migrations.

    Unlike TieredStorage's built-in migration (which is age-based),
    MigrationManager focuses on access-pattern-based promotion and
    provides configurable policies.

    Example:
        >>> manager = MigrationManager(
        ...     source=warm_store,
        ...     target=hot_store,
        ...     direction="promote",
        ...     policy=MigrationPolicy(promote_access_count=10),
        ... )
        >>> manager.register_session("session-123")
        >>> stats = await manager.run()
        >>> print(f"Promoted {stats.promoted} items")
    """

    def __init__(
        self,
        source: MemoryStore,
        target: MemoryStore,
        direction: str,  # "promote" or "demote"
        *,
        policy: MigrationPolicy | None = None,
        session_id: str | None = None,
        check_interval_seconds: int = 300,
    ) -> None:
        """Initialize MigrationManager.

        Args:
            source: Backend to migrate items FROM.
            target: Backend to migrate items TO.
            direction: "promote" (to hotter tier) or "demote" (to colder tier).
            policy: Migration policy. Defaults to MigrationPolicy().
            session_id: If provided, only migrate items from this session.
            check_interval_seconds: Interval for background checks.
        """
        if direction not in ("promote", "demote"):
            raise ValueError(
                f"direction must be 'promote' or 'demote', got {direction}"
            )

        self._source = source
        self._target = target
        self._direction = direction
        self._policy = policy or MigrationPolicy()
        self._check_interval = check_interval_seconds

        self._sessions: set[str] = set()
        if session_id:
            self._sessions.add(session_id)

        self._task: asyncio.Task[None] | None = None
        self._closed = False

        # Stats
        self._total_promoted = 0
        self._total_demoted = 0
        self._last_run: datetime | None = None

    def register_session(self, session_id: str) -> None:
        """Register a session for migration."""
        self._sessions.add(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister a session from migration."""
        self._sessions.discard(session_id)

    @property
    def total_promoted(self) -> int:
        """Total items promoted since creation."""
        return self._total_promoted

    @property
    def total_demoted(self) -> int:
        """Total items demoted since creation."""
        return self._total_demoted

    @property
    def last_run(self) -> datetime | None:
        """Timestamp of last migration run."""
        return self._last_run

    async def run(self) -> MigrationStats:
        """Run a single migration pass.

        Evaluates items in source backend against policy and migrates
        eligible items to target backend.

        Returns:
            MigrationStats with counts and timing.
        """
        start = time.perf_counter()
        stats = MigrationStats()
        self._last_run = datetime.now(UTC)
        now = self._last_run

        if not self._sessions:
            stats.duration_ms = (time.perf_counter() - start) * 1000
            return stats

        for session_id in list(self._sessions):
            try:
                keys = await self._source.list_keys(
                    session_id=session_id,
                    limit=self._policy.batch_size,
                )

                for key in keys:
                    if stats.promoted + stats.demoted >= self._policy.batch_size:
                        break

                    try:
                        metadata = await self._source.get_metadata(key)
                        if metadata is None:
                            stats.skipped += 1
                            continue

                        should_migrate = self._should_migrate(metadata, now)

                        if should_migrate:
                            success = await self._migrate_item(key)
                            if success:
                                if self._direction == "promote":
                                    stats.promoted += 1
                                    self._total_promoted += 1
                                else:
                                    stats.demoted += 1
                                    self._total_demoted += 1
                            else:
                                stats.errors += 1
                        else:
                            stats.skipped += 1

                    except Exception as e:
                        logger.warning(f"Error evaluating {key}: {e}")
                        stats.errors += 1

            except Exception as e:
                logger.error(f"Error listing keys for {session_id}: {e}")
                stats.errors += 1

        stats.duration_ms = (time.perf_counter() - start) * 1000
        return stats

    def _should_migrate(self, metadata: Any, now: datetime) -> bool:
        """Check if item should be migrated based on policy."""
        if self._direction == "promote":
            return self._should_promote(metadata, now)
        else:
            return self._should_demote(metadata, now)

    def _should_promote(self, metadata: Any, now: datetime) -> bool:
        """Check if item meets promotion criteria.

        Promotes if:
        - Access count >= threshold
        - Last access within recency window
        """
        # Check access count
        if metadata.access_count < self._policy.promote_access_count:
            return False

        # Check recency
        recency_cutoff = now - timedelta(hours=self._policy.promote_recency_hours)
        return metadata.accessed_at >= recency_cutoff

    def _should_demote(self, metadata: Any, now: datetime) -> bool:
        """Check if item meets demotion criteria.

        Demotes if:
        - Age exceeds threshold, OR
        - Low importance AND not recently accessed
        """
        age_hours = (now - metadata.created_at).total_seconds() / 3600

        # Age-based demotion
        if age_hours > self._policy.demote_age_hours:
            return True

        # Importance-based demotion (low importance items demote faster)
        if metadata.importance < self._policy.demote_min_importance:
            recency_hours = (now - metadata.accessed_at).total_seconds() / 3600
            # Demote if low importance and not accessed in half the normal window
            if recency_hours > self._policy.demote_age_hours / 2:
                return True

        return False

    async def _migrate_item(self, key: StorageKey) -> bool:
        """Migrate a single item from source to target."""
        try:
            node = await self._source.retrieve(key)
            if node is None:
                return False

            metadata = await self._source.get_metadata(key)
            await self._target.store(node, key.session_id, metadata)
            await self._source.delete(key)

            logger.debug(f"Migrated {key} ({self._direction})")
            return True

        except Exception as e:
            logger.error(f"Failed to migrate {key}: {e}")
            return False

    # Background task management

    async def start(self) -> None:
        """Start background migration task."""
        if self._task is not None:
            return

        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Started migration manager ({self._direction}) "
            f"interval={self._check_interval}s"
        )

    async def stop(self) -> None:
        """Stop background migration task."""
        if self._task is None:
            return

        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        logger.info(f"Stopped migration manager ({self._direction})")

    async def _run_loop(self) -> None:
        """Background loop for periodic migration."""
        while True:
            await asyncio.sleep(self._check_interval)
            try:
                stats = await self.run()
                if stats.promoted > 0 or stats.demoted > 0:
                    logger.info(
                        f"Migration ({self._direction}): "
                        f"promoted={stats.promoted}, demoted={stats.demoted}"
                    )
            except Exception as e:
                logger.error(f"Migration error: {e}")

    async def close(self) -> None:
        """Stop background task and cleanup."""
        if self._closed:
            return
        await self.stop()
        self._closed = True

    async def __aenter__(self) -> MigrationManager:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


@dataclass
class TierMigrationConfig:
    """Configuration for a tier pair migration."""

    source: Any  # MemoryStore
    target: Any  # MemoryStore
    direction: str  # "promote" or "demote"
    policy: MigrationPolicy = field(default_factory=MigrationPolicy)
    session_id: str | None = None


class TierMigrationCoordinator:
    """Coordinates migrations across multiple tier pairs.

    Manages multiple MigrationManagers for a complete tiered setup,
    e.g., COLD→WARM promotion and HOT→WARM demotion.

    Example:
        >>> coordinator = TierMigrationCoordinator([
        ...     TierMigrationConfig(cold_store, warm_store, "promote"),
        ...     TierMigrationConfig(hot_store, warm_store, "demote"),
        ... ])
        >>> coordinator.register_session("session-123")
        >>> await coordinator.start_all()
    """

    def __init__(
        self,
        configs: list[TierMigrationConfig],
        check_interval_seconds: int = 300,
    ) -> None:
        """Initialize coordinator with tier pair configurations."""
        self._managers: list[MigrationManager] = []

        for config in configs:
            manager = MigrationManager(
                source=config.source,
                target=config.target,
                direction=config.direction,
                policy=config.policy,
                session_id=config.session_id,
                check_interval_seconds=check_interval_seconds,
            )
            self._managers.append(manager)

    def register_session(self, session_id: str) -> None:
        """Register session with all managers."""
        for manager in self._managers:
            manager.register_session(session_id)

    def unregister_session(self, session_id: str) -> None:
        """Unregister session from all managers."""
        for manager in self._managers:
            manager.unregister_session(session_id)

    async def run_all(self) -> list[MigrationStats]:
        """Run migration on all tier pairs."""
        return [await manager.run() for manager in self._managers]

    async def start_all(self) -> None:
        """Start background tasks for all managers."""
        for manager in self._managers:
            await manager.start()

    async def stop_all(self) -> None:
        """Stop all background tasks."""
        for manager in self._managers:
            await manager.stop()

    async def close(self) -> None:
        """Close all managers."""
        for manager in self._managers:
            await manager.close()

    async def __aenter__(self) -> TierMigrationCoordinator:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
