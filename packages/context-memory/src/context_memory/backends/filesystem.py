"""FileSystemStore - Local filesystem-based storage backend.

This module provides a storage backend that persists context nodes to the
local filesystem using JSON serialization. It's suitable for development,
testing, and single-node deployments.

Directory structure:
    {base_path}/
    ├── {session_hash}/
    │   ├── nodes/
    │   │   ├── {node_id}.v{version}.json
    │   │   └── ...
    │   ├── metadata/
    │   │   ├── {node_id}.v{version}.meta.json
    │   │   └── ...
    │   └── session.json  # session_id -> hash mapping
    └── ...
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles
import aiofiles.os

from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

# Note: We use Any for node type to support testing with mock objects
# and to match the MemoryStore protocol. At runtime, nodes should be
# ContextNode instances with id, type, token_count, metadata, and model_dump().


class FileSystemStore:
    """File-based storage backend implementing MemoryStore protocol.

    This backend stores nodes and metadata as JSON files on the local filesystem.
    It supports all MemoryStore operations including batch operations, metadata
    queries, and access tracking.

    Attributes:
        base_path: Root directory for all storage.

    Example:
        >>> store = FileSystemStore("/data/context-storage")
        >>> key = await store.store(node, "session-123")
        >>> retrieved = await store.retrieve(key)
        >>> await store.close()

    Note:
        Session IDs are hashed to create safe filesystem paths. The original
        session ID is stored in session.json for reverse lookup.
    """

    def __init__(
        self,
        base_path: str | Path,
        *,
        create_if_missing: bool = True,
    ) -> None:
        """Initialize the filesystem store.

        Args:
            base_path: Root directory for storage. Will be created if it
                doesn't exist and create_if_missing is True.
            create_if_missing: If True, create base_path if it doesn't exist.

        Raises:
            FileNotFoundError: If base_path doesn't exist and create_if_missing
                is False.
        """
        self._base_path = Path(base_path)
        self._closed = False

        if create_if_missing:
            self._base_path.mkdir(parents=True, exist_ok=True)
        elif not self._base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

    @property
    def base_path(self) -> Path:
        """Get the base storage path."""
        return self._base_path

    # =========================================================================
    # Path Helpers
    # =========================================================================

    def _session_hash(self, session_id: str) -> str:
        """Create a filesystem-safe hash from session_id.

        Uses MD5 (truncated to 16 chars) for consistent, collision-resistant
        directory names. The full session_id is stored in session.json.

        Args:
            session_id: The original session identifier.

        Returns:
            16-character hex string safe for filesystem use.
        """
        return hashlib.md5(session_id.encode()).hexdigest()[:16]

    def _session_path(self, session_id: str) -> Path:
        """Get the directory path for a session.

        Args:
            session_id: Session identifier.

        Returns:
            Path to the session directory.
        """
        return self._base_path / self._session_hash(session_id)

    def _node_path(self, key: StorageKey) -> Path:
        """Get the file path for a node.

        Args:
            key: StorageKey identifying the node.

        Returns:
            Path to the node JSON file.
        """
        session_path = self._session_path(key.session_id)
        return session_path / "nodes" / f"{key.node_id}.v{key.version}.json"

    def _metadata_path(self, key: StorageKey) -> Path:
        """Get the file path for node metadata.

        Args:
            key: StorageKey identifying the node.

        Returns:
            Path to the metadata JSON file.
        """
        session_path = self._session_path(key.session_id)
        return session_path / "metadata" / f"{key.node_id}.v{key.version}.meta.json"

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    async def store(
        self,
        node: Any,
        session_id: str,
        metadata: StorageMetadata | None = None,
    ) -> StorageKey:
        """Store a context node to filesystem.

        Creates the necessary directory structure and writes both the node
        content and its metadata as separate JSON files.

        Args:
            node: The ContextNode to store.
            session_id: Session identifier for namespacing.
            metadata: Optional metadata. If not provided, auto-generated from
                node properties.

        Returns:
            StorageKey that can be used to retrieve the node.

        Raises:
            OSError: If filesystem operations fail.
        """
        # Create storage key
        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,
        )

        # Get paths
        node_path = self._node_path(key)
        metadata_path = self._metadata_path(key)

        # Ensure directories exist
        await aiofiles.os.makedirs(node_path.parent, exist_ok=True)
        await aiofiles.os.makedirs(metadata_path.parent, exist_ok=True)

        # Serialize node
        node_data = node.model_dump(mode="json")
        node_json = json.dumps(node_data, indent=2)

        # Create metadata if not provided
        if metadata is None:
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.WARM,  # Filesystem is warm tier
                size_bytes=len(node_json.encode("utf-8")),
                token_count=node.token_count or 0,
                node_type=node.type.value,
                importance=node.metadata.importance
                if hasattr(node.metadata, "importance")
                else 0.5,
                tags=set(node.metadata.tags)
                if hasattr(node.metadata, "tags")
                else set(),
            )

        # Write node file
        async with aiofiles.open(node_path, "w", encoding="utf-8") as f:
            await f.write(node_json)

        # Write metadata file
        metadata_json = json.dumps(metadata.model_dump(mode="json"), indent=2)
        async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
            await f.write(metadata_json)

        # Store session mapping
        await self._save_session_mapping(session_id)

        return key

    async def store_batch(
        self,
        nodes: list[Any],  # list[ContextNode]
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes.

        Args:
            nodes: List of ContextNodes to store.
            session_id: Session identifier for all nodes.

        Returns:
            List of StorageKeys in the same order as input nodes.
        """
        keys = []
        for node in nodes:
            key = await self.store(node, session_id)
            keys.append(key)
        return keys

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Any | None:  # ContextNode | None
        """Retrieve a node from filesystem.

        Also updates access tracking metadata (accessed_at, access_count).

        Args:
            key: The StorageKey returned from a previous store() call.

        Returns:
            The ContextNode if found, None otherwise.
        """
        node_path = self._node_path(key)

        if not node_path.exists():
            return None

        async with aiofiles.open(node_path, encoding="utf-8") as f:
            node_json = await f.read()

        node_data = json.loads(node_json)

        # Try to deserialize as ContextNode if available
        try:
            from context_core.graph import ContextNode

            node = ContextNode.model_validate(node_data)
        except ImportError:
            # Return raw dict if context-core not available (e.g., in tests)
            node = node_data

        # Update access metadata
        await self._touch_metadata(key)

        return node

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Any | None]:  # list[ContextNode | None]
        """Retrieve multiple nodes by their keys.

        Args:
            keys: List of StorageKeys to retrieve.

        Returns:
            List of ContextNodes (or None for missing keys) in the same
            order as input keys.
        """
        return [await self.retrieve(key) for key in keys]

    async def delete(
        self,
        key: StorageKey,
    ) -> bool:
        """Delete a node and its metadata.

        Args:
            key: The StorageKey of the node to delete.

        Returns:
            True if the node existed and was deleted, False if it didn't exist.
        """
        node_path = self._node_path(key)
        metadata_path = self._metadata_path(key)

        existed = node_path.exists()

        if existed:
            await aiofiles.os.remove(node_path)
        if metadata_path.exists():
            await aiofiles.os.remove(metadata_path)

        return existed

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
        return self._node_path(key).exists()

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> StorageMetadata | None:
        """Get metadata for a stored node without retrieving content.

        Args:
            key: The StorageKey to get metadata for.

        Returns:
            StorageMetadata if the key exists, None otherwise.
        """
        metadata_path = self._metadata_path(key)

        if not metadata_path.exists():
            return None

        async with aiofiles.open(metadata_path, encoding="utf-8") as f:
            metadata_json = await f.read()

        return StorageMetadata.model_validate(json.loads(metadata_json))

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict[str, Any],
    ) -> bool:
        """Update specific metadata fields for a stored node.

        Args:
            key: The StorageKey of the node to update.
            updates: Dictionary of field names to new values.

        Returns:
            True if the key existed and was updated, False if not found.
        """
        metadata = await self.get_metadata(key)
        if not metadata:
            return False

        metadata_dict = metadata.model_dump(mode="json")
        metadata_dict.update(updates)

        metadata_path = self._metadata_path(key)
        async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(metadata_dict, indent=2))

        return True

    async def _touch_metadata(self, key: StorageKey) -> None:
        """Update access time in metadata.

        Called automatically during retrieve operations.

        Args:
            key: The StorageKey of the node to touch.
        """
        metadata = await self.get_metadata(key)
        if metadata:
            metadata.touch()
            await self.update_metadata(
                key,
                {
                    "accessed_at": metadata.accessed_at.isoformat(),
                    "access_count": metadata.access_count,
                },
            )

    async def _save_session_mapping(self, session_id: str) -> None:
        """Save session_id to hash mapping for reverse lookup.

        Args:
            session_id: The original session identifier.
        """
        session_path = self._session_path(session_id)
        mapping_path = session_path / "session.json"

        if not mapping_path.exists():
            await aiofiles.os.makedirs(session_path, exist_ok=True)
            async with aiofiles.open(mapping_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps({"session_id": session_id}))

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
            node_type: If provided, only return keys for this node type.
            limit: Maximum number of keys to return.

        Returns:
            List of StorageKeys matching the criteria.
        """
        session_path = self._session_path(session_id)
        nodes_path = session_path / "nodes"

        if not nodes_path.exists():
            return []

        keys = []
        for file_path in nodes_path.glob("*.json"):
            # Parse filename: {node_id}.v{version}.json
            name = file_path.stem
            parts = name.rsplit(".v", 1)
            node_id = UUID(parts[0])
            version = int(parts[1]) if len(parts) > 1 else 1

            key = StorageKey(
                session_id=session_id,
                node_id=node_id,
                version=version,
            )

            # Apply filters if specified
            if tier or node_type:
                metadata = await self.get_metadata(key)
                if metadata:
                    if tier and metadata.tier != tier:
                        continue
                    if node_type and metadata.node_type != node_type:
                        continue

            keys.append(key)

            if len(keys) >= limit:
                break

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

        Args:
            session_id: Session to search in.
            min_importance: If provided, only return nodes with importance
                >= this value.
            tags: If provided, only return nodes that have at least one
                of these tags (OR semantics).
            since: If provided, only return nodes created after this time.
            limit: Maximum number of results.

        Returns:
            List of (StorageKey, StorageMetadata) tuples, ordered by
            importance descending.
        """
        all_keys = await self.list_keys(session_id, limit=10000)
        results: list[tuple[StorageKey, StorageMetadata]] = []

        for key in all_keys:
            metadata = await self.get_metadata(key)
            if not metadata:
                continue

            # Apply filters
            if min_importance is not None and metadata.importance < min_importance:
                continue
            if tags and not (tags & metadata.tags):
                continue
            if since and metadata.created_at < since:
                continue

            results.append((key, metadata))

            if len(results) >= limit:
                break

        # Sort by importance descending
        results.sort(key=lambda x: x[1].importance, reverse=True)

        return results

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
        """
        total_items = 0
        total_size = 0
        total_tokens = 0
        items_by_tier: dict[str, int] = {}
        size_by_tier: dict[str, int] = {}
        access_counts: list[int] = []
        oldest: datetime | None = None
        newest: datetime | None = None

        # Get sessions to process
        if session_id:
            sessions = [session_id]
        else:
            # Get all session directories
            sessions = await self._get_all_sessions()

        for sess in sessions:
            keys = await self.list_keys(sess, limit=100000)
            for key in keys:
                metadata = await self.get_metadata(key)
                if metadata:
                    total_items += 1
                    total_size += metadata.size_bytes
                    total_tokens += metadata.token_count
                    access_counts.append(metadata.access_count)

                    tier = metadata.tier.value
                    items_by_tier[tier] = items_by_tier.get(tier, 0) + 1
                    size_by_tier[tier] = size_by_tier.get(tier, 0) + metadata.size_bytes

                    if oldest is None or metadata.created_at < oldest:
                        oldest = metadata.created_at
                    if newest is None or metadata.created_at > newest:
                        newest = metadata.created_at

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

    async def _get_all_sessions(self) -> list[str]:
        """Get all session IDs from stored mappings.

        Returns:
            List of session IDs with stored data.
        """
        sessions = []
        for dir_path in self._base_path.iterdir():
            if dir_path.is_dir():
                mapping_path = dir_path / "session.json"
                if mapping_path.exists():
                    async with aiofiles.open(mapping_path, encoding="utf-8") as f:
                        data = json.loads(await f.read())
                        sessions.append(data["session_id"])
        return sessions

    async def close(self) -> None:
        """Close the store and release resources.

        For filesystem store, this is a no-op but provided for protocol
        compliance. Safe to call multiple times.
        """
        self._closed = True
