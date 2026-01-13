"""RedisStore - Redis-based hot-tier storage backend.

This module provides a high-performance storage backend using Redis for
hot-tier caching. Optimized for fast access with automatic TTL-based
expiration.

Features:
    - Sub-10ms latency for reads and writes
    - Automatic TTL-based expiration (default 1 hour)
    - Access resets TTL (touch on read)
    - Pipeline support for batch operations
    - Efficient SCAN-based key iteration

Key Structure:
    ctx:{session_id}:{node_id}:v{version} -> Hash with fields:
        - node_data: JSON serialized node
        - tier, node_type, importance, size_bytes, token_count
        - tags: JSON array
        - created_at, accessed_at, access_count

Usage:
    Hot tier in tiered storage architecture with < 1 hour retention.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

if TYPE_CHECKING:
    import redis.asyncio as redis


# Default TTL: 1 hour (hot tier retention)
DEFAULT_TTL_SECONDS = 3600


class RedisStore:
    """Redis storage backend for hot-tier caching.

    This backend uses Redis for high-performance, low-latency storage
    with automatic TTL-based expiration. Ideal for frequently accessed
    context that needs sub-10ms access times.

    Attributes:
        url: Redis connection URL.
        ttl_seconds: Time-to-live for stored items.
        key_prefix: Prefix for all Redis keys.

    Example:
        >>> store = RedisStore("redis://localhost:6379")
        >>> await store.initialize()
        >>> key = await store.store(node, "session-123")
        >>> retrieved = await store.retrieve(key)  # Resets TTL
        >>> await store.close()

    Note:
        Requires the 'redis' optional dependency:
        pip install context-memory[redis]
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = "ctx",
        db: int = 0,
    ) -> None:
        """Initialize the Redis store.

        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379").
            ttl_seconds: TTL for stored items in seconds. Default 1 hour.
            key_prefix: Prefix for all Redis keys. Default "ctx".
            db: Redis database number. Default 0.
        """
        self._url = url
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        self._db = db
        self._client: redis.Redis | None = None
        self._initialized = False

    def _make_key(self, key: StorageKey) -> str:
        """Create Redis key from StorageKey."""
        return f"{self._key_prefix}:{key.session_id}:{key.node_id}:v{key.version}"

    def _parse_key(self, redis_key: str) -> StorageKey | None:
        """Parse Redis key back to StorageKey."""
        try:
            parts = redis_key.split(":")
            if len(parts) < 4:
                return None
            # Format: prefix:session_id:node_id:vN
            session_id = parts[1]
            node_id = parts[2]
            version_str = parts[3]
            version = int(version_str[1:]) if version_str.startswith("v") else 1
            from uuid import UUID

            return StorageKey(
                session_id=session_id,
                node_id=UUID(node_id),
                version=version,
            )
        except (ValueError, IndexError):
            return None

    @property
    def _redis(self) -> redis.Redis:
        """Get Redis client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Call initialize() first.")
        return self._client

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize Redis connection.

        Creates the Redis client connection. Safe to call multiple times.
        """
        if self._initialized:
            return

        try:
            import redis.asyncio as redis_lib
        except ImportError as e:
            raise ImportError(
                "redis is required for RedisStore. "
                "Install with: pip install context-memory[redis]"
            ) from e

        self._client = redis_lib.from_url(
            self._url,
            db=self._db,
            decode_responses=True,
        )

        # Test connection
        await self._client.ping()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        """Ensure the store is initialized before operations."""
        if not self._initialized:
            await self.initialize()

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(
        self,
        node: Any,
        session_id: str,
        metadata: StorageMetadata | None = None,
    ) -> StorageKey:
        """Store a context node with TTL.

        Args:
            node: The ContextNode to store (or compatible mock object).
            session_id: Session identifier for namespacing.
            metadata: Optional metadata. If not provided, auto-generated.

        Returns:
            StorageKey for retrieval.
        """
        await self._ensure_initialized()

        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,
        )

        # Serialize node
        node_data = json.dumps(node.model_dump(mode="json"))
        now = datetime.now(UTC)

        # Create metadata if not provided
        if metadata is None:
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.HOT,  # Redis is hot tier
                size_bytes=len(node_data.encode("utf-8")),
                token_count=node.token_count or 0,
                node_type=node.type.value,
                importance=node.metadata.importance
                if hasattr(node.metadata, "importance")
                else 0.5,
                tags=set(node.metadata.tags)
                if hasattr(node.metadata, "tags")
                else set(),
            )

        redis_key = self._make_key(key)

        # Store as hash for atomic access
        hash_data = {
            "node_data": node_data,
            "tier": metadata.tier.value,
            "node_type": metadata.node_type,
            "importance": str(metadata.importance),
            "size_bytes": str(metadata.size_bytes),
            "token_count": str(metadata.token_count),
            "tags": json.dumps(list(metadata.tags)),
            "created_at": now.isoformat(),
            "accessed_at": now.isoformat(),
            "access_count": "0",
        }

        # Use pipeline for atomic set + expire
        async with self._redis.pipeline() as pipe:
            pipe.hset(redis_key, mapping=hash_data)
            pipe.expire(redis_key, self._ttl_seconds)
            await pipe.execute()

        return key

    async def store_batch(
        self,
        nodes: list[Any],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes efficiently using pipeline.

        Args:
            nodes: List of ContextNodes to store.
            session_id: Session identifier for all nodes.

        Returns:
            List of StorageKeys in the same order as input nodes.
        """
        await self._ensure_initialized()

        keys = []
        now = datetime.now(UTC)

        async with self._redis.pipeline() as pipe:
            for node in nodes:
                key = StorageKey(
                    session_id=session_id,
                    node_id=node.id,
                    version=1,
                )
                keys.append(key)

                node_data = json.dumps(node.model_dump(mode="json"))
                redis_key = self._make_key(key)

                token_count = node.token_count or 0
                importance = (
                    node.metadata.importance
                    if hasattr(node.metadata, "importance")
                    else 0.5
                )
                tags = (
                    list(node.metadata.tags) if hasattr(node.metadata, "tags") else []
                )

                hash_data = {
                    "node_data": node_data,
                    "tier": StorageTier.HOT.value,
                    "node_type": node.type.value,
                    "importance": str(importance),
                    "size_bytes": str(len(node_data.encode("utf-8"))),
                    "token_count": str(token_count),
                    "tags": json.dumps(tags),
                    "created_at": now.isoformat(),
                    "accessed_at": now.isoformat(),
                    "access_count": "0",
                }

                pipe.hset(redis_key, mapping=hash_data)
                pipe.expire(redis_key, self._ttl_seconds)

            await pipe.execute()

        return keys

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Any | None:
        """Retrieve a node by key, resetting TTL.

        Access resets the TTL (touch on read) to keep frequently
        accessed items in the hot tier.

        Args:
            key: StorageKey from previous store operation.

        Returns:
            The node if found, None otherwise.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)

        # Get data and update access tracking atomically
        async with self._redis.pipeline() as pipe:
            pipe.hgetall(redis_key)
            pipe.hincrby(redis_key, "access_count", 1)
            pipe.hset(redis_key, "accessed_at", datetime.now(UTC).isoformat())
            pipe.expire(redis_key, self._ttl_seconds)  # Reset TTL
            results = await pipe.execute()

        data = results[0]
        if not data:
            return None

        node_data = json.loads(data["node_data"])

        # Try to deserialize as ContextNode if available
        try:
            from context_core.graph import ContextNode

            return ContextNode.model_validate(node_data)
        except ImportError:
            return node_data
        except Exception:
            return node_data

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Any | None]:
        """Retrieve multiple nodes by their keys.

        Args:
            keys: List of StorageKeys to retrieve.

        Returns:
            List of nodes (or None for missing keys) in the same order.
        """
        await self._ensure_initialized()

        if not keys:
            return []

        now = datetime.now(UTC).isoformat()

        # Use pipeline for batch retrieval
        async with self._redis.pipeline() as pipe:
            for key in keys:
                redis_key = self._make_key(key)
                pipe.hgetall(redis_key)

            results = await pipe.execute()

        # Update access tracking for found keys
        async with self._redis.pipeline() as pipe:
            for key, data in zip(keys, results, strict=False):
                if data:
                    redis_key = self._make_key(key)
                    pipe.hincrby(redis_key, "access_count", 1)
                    pipe.hset(redis_key, "accessed_at", now)
                    pipe.expire(redis_key, self._ttl_seconds)
            await pipe.execute()

        # Parse results
        context_node_cls = None
        try:
            from context_core.graph import ContextNode

            context_node_cls = ContextNode
        except ImportError:
            pass

        parsed = []
        for data in results:
            if not data:
                parsed.append(None)
                continue

            node_data = json.loads(data["node_data"])
            if context_node_cls:
                try:
                    parsed.append(context_node_cls.model_validate(node_data))
                except Exception:
                    parsed.append(node_data)
            else:
                parsed.append(node_data)

        return parsed

    async def delete(
        self,
        key: StorageKey,
    ) -> bool:
        """Delete a node.

        Args:
            key: StorageKey of node to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)
        deleted = await self._redis.delete(redis_key)
        return deleted > 0

    async def exists(
        self,
        key: StorageKey,
    ) -> bool:
        """Check if a key exists.

        Args:
            key: The StorageKey to check.

        Returns:
            True if the key exists, False otherwise.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)
        return await self._redis.exists(redis_key) > 0

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> StorageMetadata | None:
        """Get metadata without retrieving node content.

        Does NOT reset TTL (metadata-only access).

        Args:
            key: StorageKey to get metadata for.

        Returns:
            StorageMetadata if found, None otherwise.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)

        # Get only metadata fields (not node_data)
        # Note: redis-py type stubs show Awaitable[X] | X for sync/async API
        data = await self._redis.hgetall(redis_key)  # type: ignore[misc]

        if not data:
            return None

        return StorageMetadata(
            key=key,
            tier=StorageTier(data["tier"]),
            node_type=data["node_type"],
            importance=float(data["importance"]),
            size_bytes=int(data["size_bytes"]),
            token_count=int(data["token_count"]),
            tags=set(json.loads(data["tags"])),
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=int(data["access_count"]),
        )

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict[str, Any],
    ) -> bool:
        """Update specific metadata fields.

        Args:
            key: StorageKey of the node to update.
            updates: Dictionary of field names to new values.

        Returns:
            True if updated, False if not found.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)

        # Check if key exists
        if not await self._redis.exists(redis_key):
            return False

        # Map field names to hash fields
        hash_updates = {}
        for field, value in updates.items():
            if field == "tier":
                if isinstance(value, StorageTier):
                    value = value.value
                hash_updates["tier"] = value
            elif field == "importance":
                hash_updates["importance"] = str(value)
            elif field == "tags":
                if isinstance(value, set):
                    value = list(value)
                hash_updates["tags"] = json.dumps(value)
            elif field == "access_count":
                hash_updates["access_count"] = str(value)
            elif field == "accessed_at":
                if isinstance(value, datetime):
                    value = value.isoformat()
                hash_updates["accessed_at"] = value

        if hash_updates:
            await self._redis.hset(redis_key, mapping=hash_updates)  # type: ignore[misc]

        return True

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def list_keys(
        self,
        session_id: str,
        *,
        tier: StorageTier | None = None,
        node_type: str | None = None,
        limit: int = 1000,
    ) -> list[StorageKey]:
        """List storage keys for a session.

        Uses SCAN for efficient iteration over large keyspaces.

        Args:
            session_id: Session to list keys for.
            tier: If provided, filter by storage tier.
            node_type: If provided, filter by node type.
            limit: Maximum number of keys to return.

        Returns:
            List of StorageKeys matching the criteria.
        """
        await self._ensure_initialized()

        pattern = f"{self._key_prefix}:{session_id}:*"
        keys: list[StorageKey] = []

        async for redis_key in self._redis.scan_iter(match=pattern, count=100):
            if len(keys) >= limit:
                break

            storage_key = self._parse_key(redis_key)
            if storage_key is None:
                continue

            # Apply filters if specified
            if tier is not None or node_type is not None:
                data = await self._redis.hmget(redis_key, ["tier", "node_type"])  # type: ignore[misc]
                if tier is not None and data[0] != tier.value:
                    continue
                if node_type is not None and data[1] != node_type:
                    continue

            keys.append(storage_key)

        return keys

    async def search_by_metadata(
        self,
        session_id: str,
        *,
        min_importance: float | None = None,
        tags: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        """Search for nodes by metadata criteria.

        Note: This requires scanning keys, which may be slow for large
        datasets. Consider using PostgreSQL/SQLite for complex queries.

        Args:
            session_id: Session to search in.
            min_importance: Minimum importance threshold.
            tags: Tags to match (OR semantics).
            since: Only return nodes created after this time.
            limit: Maximum number of results.

        Returns:
            List of (StorageKey, StorageMetadata) tuples, ordered by
            importance descending.
        """
        await self._ensure_initialized()

        pattern = f"{self._key_prefix}:{session_id}:*"
        results: list[tuple[StorageKey, StorageMetadata]] = []

        async for redis_key in self._redis.scan_iter(match=pattern, count=100):
            storage_key = self._parse_key(redis_key)
            if storage_key is None:
                continue

            data = await self._redis.hgetall(redis_key)  # type: ignore[misc]
            if not data:
                continue

            importance = float(data["importance"])
            created_at = datetime.fromisoformat(data["created_at"])
            node_tags = set(json.loads(data["tags"]))

            # Apply filters
            if min_importance is not None and importance < min_importance:
                continue
            if tags is not None and not (tags & node_tags):
                continue
            if since is not None and created_at < since:
                continue

            metadata = StorageMetadata(
                key=storage_key,
                tier=StorageTier(data["tier"]),
                node_type=data["node_type"],
                importance=importance,
                size_bytes=int(data["size_bytes"]),
                token_count=int(data["token_count"]),
                tags=node_tags,
                created_at=created_at,
                accessed_at=datetime.fromisoformat(data["accessed_at"]),
                access_count=int(data["access_count"]),
            )

            results.append((storage_key, metadata))

        # Sort by importance descending
        results.sort(key=lambda x: x[1].importance, reverse=True)

        return results[:limit]

    # =========================================================================
    # Statistics & Lifecycle
    # =========================================================================

    async def stats(
        self,
        session_id: str | None = None,
    ) -> StorageStats:
        """Get storage statistics.

        Args:
            session_id: If provided, stats for this session only.
                If None, aggregate stats across all sessions.

        Returns:
            StorageStats with counts, sizes, and tier breakdowns.
        """
        await self._ensure_initialized()

        if session_id:
            pattern = f"{self._key_prefix}:{session_id}:*"
        else:
            pattern = f"{self._key_prefix}:*"

        total_items = 0
        total_size = 0
        total_tokens = 0
        access_counts: list[int] = []
        items_by_tier: dict[str, int] = {}
        size_by_tier: dict[str, int] = {}
        oldest: datetime | None = None
        newest: datetime | None = None

        async for redis_key in self._redis.scan_iter(match=pattern, count=100):
            data = await self._redis.hgetall(redis_key)  # type: ignore[misc]
            if not data:
                continue

            total_items += 1
            size = int(data["size_bytes"])
            total_size += size
            total_tokens += int(data["token_count"])
            access_counts.append(int(data["access_count"]))

            tier = data["tier"]
            items_by_tier[tier] = items_by_tier.get(tier, 0) + 1
            size_by_tier[tier] = size_by_tier.get(tier, 0) + size

            created = datetime.fromisoformat(data["created_at"])
            if oldest is None or created < oldest:
                oldest = created
            if newest is None or created > newest:
                newest = created

        return StorageStats(
            total_items=total_items,
            total_size_bytes=total_size,
            total_tokens=total_tokens,
            items_by_tier=items_by_tier,
            size_by_tier=size_by_tier,
            avg_access_count=(
                sum(access_counts) / len(access_counts) if access_counts else 0.0
            ),
            oldest_item=oldest,
            newest_item=newest,
        )

    async def close(self) -> None:
        """Close Redis connection.

        Safe to call multiple times.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    # =========================================================================
    # Redis-Specific Operations
    # =========================================================================

    async def touch(self, key: StorageKey) -> bool:
        """Reset TTL for a key without retrieving content.

        Useful for keeping items in the hot tier.

        Args:
            key: StorageKey to touch.

        Returns:
            True if key exists and was touched, False otherwise.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)

        if not await self._redis.exists(redis_key):
            return False

        async with self._redis.pipeline() as pipe:
            pipe.hset(redis_key, "accessed_at", datetime.now(UTC).isoformat())
            pipe.hincrby(redis_key, "access_count", 1)
            pipe.expire(redis_key, self._ttl_seconds)
            await pipe.execute()

        return True

    async def get_ttl(self, key: StorageKey) -> int | None:
        """Get remaining TTL for a key in seconds.

        Args:
            key: StorageKey to check.

        Returns:
            Remaining TTL in seconds, or None if key doesn't exist.
            Returns -1 if key has no TTL.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)
        ttl = await self._redis.ttl(redis_key)

        if ttl == -2:  # Key doesn't exist
            return None
        return ttl

    async def set_ttl(self, key: StorageKey, ttl_seconds: int) -> bool:
        """Set TTL for a specific key.

        Args:
            key: StorageKey to update.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if TTL was set, False if key doesn't exist.
        """
        await self._ensure_initialized()

        redis_key = self._make_key(key)
        return await self._redis.expire(redis_key, ttl_seconds)

    async def flush_session(self, session_id: str) -> int:
        """Delete all keys for a session.

        Args:
            session_id: Session to flush.

        Returns:
            Number of keys deleted.
        """
        await self._ensure_initialized()

        pattern = f"{self._key_prefix}:{session_id}:*"
        deleted = 0

        async for redis_key in self._redis.scan_iter(match=pattern, count=100):
            await self._redis.delete(redis_key)
            deleted += 1

        return deleted
