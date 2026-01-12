"""MemoryStore protocol definition for context-memory package.

This module defines the contract that all storage backends must implement.
The protocol is designed to be async-first to support various backends
including databases, Redis, S3, and filesystem storage.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

if TYPE_CHECKING:
    from context_core.graph import ContextNode


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for context memory storage.

    All methods are async to support various backends (databases, Redis, S3, etc.).
    Implementations should handle their own connection pooling and resource management.

    This protocol is @runtime_checkable, allowing isinstance() checks:

        >>> isinstance(my_store, MemoryStore)
        True

    Example implementation:

        class FileSystemStore:
            async def store(self, node, session_id, metadata=None):
                # Implementation here
                ...

            # ... implement all other methods
    """

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(
        self,
        node: ContextNode,
        session_id: str,
        metadata: StorageMetadata | None = None,
    ) -> StorageKey:
        """Store a context node.

        Args:
            node: The ContextNode to store.
            session_id: Session identifier for namespacing. Nodes are isolated
                by session to support multi-tenancy.
            metadata: Optional metadata. If not provided, metadata will be
                auto-generated from the node's properties (type, importance,
                tags, token_count).

        Returns:
            StorageKey that can be used to retrieve the node.

        Raises:
            StorageError: If the storage operation fails.

        Example:
            >>> key = await store.store(node, "session-123")
            >>> print(key)
            session-123/550e8400-e29b-41d4-a716-446655440000/1
        """
        ...

    async def store_batch(
        self,
        nodes: list[Any],  # list[ContextNode] - Any to avoid import
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes efficiently.

        Implementations should optimize for batch operations (e.g., using
        database transactions or bulk inserts).

        Args:
            nodes: List of ContextNodes to store.
            session_id: Session identifier for all nodes.

        Returns:
            List of StorageKeys in the same order as input nodes.

        Raises:
            StorageError: If any storage operation fails. Implementations
                should be atomic where possible (all-or-nothing).
        """
        ...

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Any | None:  # Optional[ContextNode]
        """Retrieve a node by its storage key.

        This method should also update access tracking metadata (accessed_at,
        access_count) to support LRU eviction and tier migration.

        Args:
            key: The StorageKey returned from a previous store() call.

        Returns:
            The ContextNode if found, None otherwise.

        Example:
            >>> node = await store.retrieve(key)
            >>> if node is None:
            ...     print("Node not found or expired")
        """
        ...

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Any | None]:  # list[Optional[ContextNode]]
        """Retrieve multiple nodes by their keys.

        Args:
            keys: List of StorageKeys to retrieve.

        Returns:
            List of ContextNodes (or None for missing keys) in the same
            order as input keys. The returned list will always have the
            same length as the input keys list.

        Example:
            >>> nodes = await store.retrieve_batch([key1, key2, key3])
            >>> # nodes[i] corresponds to keys[i], may be None if not found
        """
        ...

    async def delete(
        self,
        key: StorageKey,
    ) -> bool:
        """Delete a node by its storage key.

        Args:
            key: The StorageKey of the node to delete.

        Returns:
            True if the node existed and was deleted, False if it didn't exist.

        Note:
            Implementations should also clean up associated metadata.
        """
        ...

    async def exists(
        self,
        key: StorageKey,
    ) -> bool:
        """Check if a key exists without retrieving the full node.

        This is more efficient than retrieve() when you only need to
        check existence.

        Args:
            key: The StorageKey to check.

        Returns:
            True if the key exists, False otherwise.
        """
        ...

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> StorageMetadata | None:
        """Get metadata for a stored node without retrieving content.

        Useful for making decisions about tier migration, eviction, or
        filtering without the cost of deserializing the full node.

        Args:
            key: The StorageKey to get metadata for.

        Returns:
            StorageMetadata if the key exists, None otherwise.
        """
        ...

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict[str, Any],
    ) -> bool:
        """Update specific metadata fields for a stored node.

        Args:
            key: The StorageKey of the node to update.
            updates: Dictionary of field names to new values. Only the
                specified fields will be updated; others remain unchanged.

        Returns:
            True if the key existed and was updated, False if not found.

        Example:
            >>> await store.update_metadata(key, {
            ...     "importance": 0.9,
            ...     "tags": {"important", "reviewed"},
            ... })
        """
        ...

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
        """List storage keys for a session with optional filters.

        Args:
            session_id: Session to list keys for.
            tier: If provided, only return keys in this storage tier.
            node_type: If provided, only return keys for this node type
                (e.g., "MESSAGE", "TOOL_CALL").
            limit: Maximum number of keys to return. Default 1000.

        Returns:
            List of StorageKeys matching the criteria.

        Note:
            Results are not guaranteed to be in any particular order.
            Use search_by_metadata() for ordered results.
        """
        ...

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

        This is more flexible than list_keys() and returns metadata
        along with keys to avoid additional lookups.

        Args:
            session_id: Session to search in.
            min_importance: If provided, only return nodes with importance
                >= this value.
            tags: If provided, only return nodes that have at least one
                of these tags (OR semantics).
            since: If provided, only return nodes created after this time.
            limit: Maximum number of results. Default 100.

        Returns:
            List of (StorageKey, StorageMetadata) tuples, typically ordered
            by importance descending.

        Example:
            >>> results = await store.search_by_metadata(
            ...     "session-123",
            ...     min_importance=0.7,
            ...     tags={"tool_result", "important"},
            ...     limit=50,
            ... )
            >>> for key, metadata in results:
            ...     print(f"{key}: importance={metadata.importance}")
        """
        ...

    # =========================================================================
    # Statistics & Lifecycle
    # =========================================================================

    async def stats(
        self,
        session_id: str | None = None,
    ) -> StorageStats:
        """Get storage statistics.

        Args:
            session_id: If provided, return stats for this session only.
                If None, return aggregate stats across all sessions.

        Returns:
            StorageStats with counts, sizes, and tier breakdowns.

        Example:
            >>> stats = await store.stats("session-123")
            >>> print(f"Total items: {stats.total_items}")
            >>> print(f"Hot tier: {stats.items_by_tier.get('hot', 0)}")
        """
        ...

    async def close(self) -> None:
        """Close connections and release resources.

        Implementations should:
        - Close database connections/connection pools
        - Flush any pending writes
        - Release file handles
        - Cancel background tasks

        This method should be safe to call multiple times.

        Example:
            >>> store = FileSystemStore("/data/context")
            >>> try:
            ...     # use store
            ... finally:
            ...     await store.close()
        """
        ...
