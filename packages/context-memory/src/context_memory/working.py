"""WorkingMemory - Fast-access LRU cache for active context.

Provides an in-memory cache layer with:
- LRU eviction based on token and item limits
- Background sync to persistent backing store
- Dirty tracking for efficient persistence
- Token-aware capacity management
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any
from uuid import UUID

from context_memory.types import StorageKey, StorageMetadata, StorageTier

if TYPE_CHECKING:
    from context_memory.store import MemoryStore


class WorkingMemoryStats:
    """Statistics for WorkingMemory monitoring."""

    def __init__(
        self,
        items: int,
        tokens: int,
        max_tokens: int,
        max_items: int,
        dirty_items: int,
    ) -> None:
        self.items = items
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.max_items = max_items
        self.dirty_items = dirty_items

    @property
    def token_utilization(self) -> float:
        """Token capacity utilization (0-1)."""
        return self.tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    @property
    def item_utilization(self) -> float:
        """Item capacity utilization (0-1)."""
        return self.items / self.max_items if self.max_items > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items": self.items,
            "tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "max_items": self.max_items,
            "dirty_items": self.dirty_items,
            "token_utilization": self.token_utilization,
            "item_utilization": self.item_utilization,
        }


class WorkingMemory:
    """Fast-access LRU cache for active context nodes.

    Provides a high-performance in-memory cache with automatic eviction
    and background persistence to a backing store. Designed to hold the
    most recently accessed context for quick retrieval.

    Features:
        - **LRU Eviction**: Least-recently-used items evicted when capacity exceeded
        - **Token-aware**: Respects token limits for context budget management
        - **Background Sync**: Periodic persistence of dirty items
        - **Dirty Tracking**: Only persists modified items

    Example:
        >>> from context_memory.backends import FileSystemStore
        >>> store = FileSystemStore("/tmp/context")
        >>> memory = WorkingMemory(backing_store=store, max_tokens=10000)
        >>> await memory.start()
        >>> await memory.add(node, session_id="sess-123")
        >>> node = await memory.get(node_id)
        >>> await memory.stop()

    Note:
        Always call `start()` to begin background sync and `stop()` to
        ensure dirty items are flushed before shutdown.
    """

    def __init__(
        self,
        backing_store: MemoryStore,
        max_tokens: int = 50000,
        max_items: int = 1000,
        sync_interval_seconds: int = 60,
    ) -> None:
        """Initialize WorkingMemory.

        Args:
            backing_store: Persistent store for evicted/synced items
            max_tokens: Maximum total tokens to hold in memory
            max_items: Maximum number of items in cache
            sync_interval_seconds: Interval between background syncs
        """
        self._backing_store = backing_store
        self._max_tokens = max_tokens
        self._max_items = max_items
        self._sync_interval = sync_interval_seconds

        # LRU cache: OrderedDict maintains insertion order, move_to_end for access
        self._cache: OrderedDict[UUID, Any] = OrderedDict()  # Any = ContextNode
        self._metadata: dict[UUID, StorageMetadata] = {}
        self._dirty: set[UUID] = set()  # Modified but not yet synced
        self._current_tokens = 0

        self._sync_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Start background sync task.

        Should be called after initialization to begin periodic
        persistence of dirty items.
        """
        if self._sync_task is not None:
            return
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop(self) -> None:
        """Stop background sync and flush all dirty items.

        Should be called before shutdown to ensure all modified
        items are persisted.
        """
        if self._sync_task is not None:
            self._sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task
            self._sync_task = None

        # Final flush
        await self.flush()
        self._closed = True

    async def __aenter__(self) -> WorkingMemory:
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.stop()

    # =========================================================================
    # Core Operations
    # =========================================================================

    async def add(
        self,
        node: Any,  # ContextNode
        session_id: str,
    ) -> None:
        """Add a node to working memory.

        If adding the node would exceed capacity, LRU items are evicted
        first. Evicted items are persisted to the backing store if dirty.

        Args:
            node: ContextNode to add
            session_id: Session the node belongs to
        """
        async with self._lock:
            node_tokens = getattr(node, "token_count", 0) or 0

            # Evict if needed to make room
            while (
                self._current_tokens + node_tokens > self._max_tokens
                or len(self._cache) >= self._max_items
            ):
                if not self._cache:
                    break
                await self._evict_one()

            # Add to cache (at end = most recently used)
            self._cache[node.id] = node
            self._cache.move_to_end(node.id)
            self._current_tokens += node_tokens
            self._dirty.add(node.id)

            # Create metadata for this node
            node_type = getattr(node.type, "value", str(node.type))
            importance = getattr(getattr(node, "metadata", None), "importance", 0.5)
            tags = getattr(getattr(node, "metadata", None), "tags", set()) or set()

            self._metadata[node.id] = StorageMetadata(
                key=StorageKey(session_id=session_id, node_id=node.id, version=1),
                tier=StorageTier.HOT,
                size_bytes=0,
                token_count=node_tokens,
                node_type=node_type,
                importance=importance,
                tags=tags,
            )

    async def get(self, node_id: UUID) -> Any | None:
        """Get a node from working memory.

        Accessing a node moves it to the end of the LRU queue,
        protecting it from eviction.

        Args:
            node_id: ID of node to retrieve

        Returns:
            ContextNode if found, None otherwise
        """
        async with self._lock:
            node = self._cache.get(node_id)
            if node is not None:
                # Move to end (most recently used)
                self._cache.move_to_end(node_id)
                # Update access tracking
                if node_id in self._metadata:
                    self._metadata[node_id].touch()
            return node

    async def remove(self, node_id: UUID) -> Any | None:
        """Remove a node from working memory.

        Does NOT persist to backing store. Use `flush()` first if
        persistence is needed.

        Args:
            node_id: ID of node to remove

        Returns:
            Removed node if found, None otherwise
        """
        async with self._lock:
            node = self._cache.pop(node_id, None)
            if node is not None:
                node_tokens = getattr(node, "token_count", 0) or 0
                self._current_tokens -= node_tokens
                self._metadata.pop(node_id, None)
                self._dirty.discard(node_id)
            return node

    async def contains(self, node_id: UUID) -> bool:
        """Check if a node is in working memory.

        Args:
            node_id: ID to check

        Returns:
            True if node is cached
        """
        return node_id in self._cache

    async def clear(self) -> int:
        """Clear all items from working memory.

        Does NOT persist dirty items. Call `flush()` first if needed.

        Returns:
            Number of items cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._metadata.clear()
            self._dirty.clear()
            self._current_tokens = 0
            return count

    # =========================================================================
    # Eviction
    # =========================================================================

    async def _evict_one(self) -> None:
        """Evict the least recently used item.

        If the item is dirty, it is persisted to the backing store
        before removal.
        """
        if not self._cache:
            return

        # Get LRU item (first in OrderedDict)
        node_id, node = next(iter(self._cache.items()))

        # Persist if dirty before evicting
        if node_id in self._dirty:
            metadata = self._metadata.get(node_id)
            if metadata is not None:
                await self._backing_store.store(
                    node,
                    metadata.key.session_id,
                    metadata,
                )
            self._dirty.discard(node_id)

        # Remove from cache
        del self._cache[node_id]
        self._metadata.pop(node_id, None)
        node_tokens = getattr(node, "token_count", 0) or 0
        self._current_tokens -= node_tokens

    # =========================================================================
    # Persistence
    # =========================================================================

    async def flush(self) -> int:
        """Persist all dirty items to backing store.

        Called automatically by background sync task, but can be
        called manually for immediate persistence.

        Returns:
            Number of items flushed
        """
        async with self._lock:
            flushed = 0
            dirty_ids = list(self._dirty)

            for node_id in dirty_ids:
                node = self._cache.get(node_id)
                metadata = self._metadata.get(node_id)

                if node is not None and metadata is not None:
                    await self._backing_store.store(
                        node,
                        metadata.key.session_id,
                        metadata,
                    )
                    flushed += 1

            self._dirty.clear()
            return flushed

    async def _sync_loop(self) -> None:
        """Background loop for periodic sync."""
        while True:
            await asyncio.sleep(self._sync_interval)
            with contextlib.suppress(Exception):
                await self.flush()

    # =========================================================================
    # Loading from Store
    # =========================================================================

    async def load_from_store(
        self,
        session_id: str,
        limit: int = 100,
    ) -> int:
        """Load recent items from backing store into cache.

        Useful for warming the cache after startup.

        Args:
            session_id: Session to load items from
            limit: Maximum items to load

        Returns:
            Number of items loaded
        """
        keys = await self._backing_store.list_keys(session_id=session_id, limit=limit)
        loaded = 0

        for key in keys:
            if len(self._cache) >= self._max_items:
                break

            # Skip if already in cache
            if key.node_id in self._cache:
                continue

            node = await self._backing_store.retrieve(key)
            if node is None:
                continue

            node_tokens = getattr(node, "token_count", 0) or 0

            # Check capacity
            if self._current_tokens + node_tokens > self._max_tokens:
                continue

            async with self._lock:
                self._cache[node.id] = node
                self._current_tokens += node_tokens

                metadata = await self._backing_store.get_metadata(key)
                if metadata is not None:
                    self._metadata[node.id] = metadata

                # Not dirty since loaded from store
                loaded += 1

        return loaded

    # =========================================================================
    # Statistics
    # =========================================================================

    @property
    def stats(self) -> WorkingMemoryStats:
        """Get cache statistics.

        Returns:
            WorkingMemoryStats with current state
        """
        return WorkingMemoryStats(
            items=len(self._cache),
            tokens=self._current_tokens,
            max_tokens=self._max_tokens,
            max_items=self._max_items,
            dirty_items=len(self._dirty),
        )

    @property
    def item_count(self) -> int:
        """Number of items in cache."""
        return len(self._cache)

    @property
    def token_count(self) -> int:
        """Total tokens in cache."""
        return self._current_tokens

    @property
    def dirty_count(self) -> int:
        """Number of dirty (unsynced) items."""
        return len(self._dirty)
