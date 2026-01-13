"""SQLiteStore - SQLite-based storage backend.

This module provides a lightweight, embedded storage backend using SQLite
with async support via aiosqlite. Ideal for single-process deployments,
development, and moderate workloads.

Performance optimizations:
- WAL mode for better concurrent read/write
- Proper indexing for common query patterns
- Batch operations using transactions
- JSON1 extension for metadata queries
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiosqlite

from context_memory.types import (
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

# Schema SQL
_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_nodes (
    -- Composite primary key for versioning
    node_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    -- Data storage
    node_data TEXT NOT NULL,
    -- Denormalized metadata for fast queries
    tier TEXT NOT NULL DEFAULT 'warm',
    node_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    size_bytes INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    -- Timestamps
    created_at TEXT NOT NULL,
    accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    -- Full metadata JSON for complex queries
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (node_id, session_id, version)
);

CREATE INDEX IF NOT EXISTS idx_session
    ON context_nodes(session_id);

CREATE INDEX IF NOT EXISTS idx_session_tier
    ON context_nodes(session_id, tier);

CREATE INDEX IF NOT EXISTS idx_session_type
    ON context_nodes(session_id, node_type);

CREATE INDEX IF NOT EXISTS idx_session_importance
    ON context_nodes(session_id, importance DESC);

CREATE INDEX IF NOT EXISTS idx_session_created
    ON context_nodes(session_id, created_at DESC);
"""


class SQLiteStore:
    """SQLite-based storage backend implementing MemoryStore protocol.

    Uses aiosqlite for async operations with a single-file database.
    Optimized for single-process deployments with moderate workloads.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> store = SQLiteStore("/data/context.db")
        >>> await store.initialize()
        >>> key = await store.store(node, "session-123")
        >>> await store.close()

    Performance notes:
        - Uses WAL mode for better concurrency
        - Batch operations wrapped in transactions
        - Denormalized columns for fast filtering
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        wal_mode: bool = True,
    ) -> None:
        """Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file. Use ":memory:" for in-memory.
            wal_mode: Enable WAL mode for better concurrency. Default True.
        """
        self._db_path = str(db_path)
        self._wal_mode = wal_mode
        self._conn: aiosqlite.Connection | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection and schema.

        Must be called before any operations. Safe to call multiple times.
        """
        if self._initialized:
            return

        self._conn = await aiosqlite.connect(self._db_path)

        # Enable WAL mode for better concurrent access
        if self._wal_mode and self._db_path != ":memory:":
            await self._db.execute("PRAGMA journal_mode=WAL")

        # Performance pragmas
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA cache_size=-64000")  # 64MB cache
        await self._db.execute("PRAGMA temp_store=MEMORY")

        # Create schema
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

        self._initialized = True

    @property
    def _db(self) -> aiosqlite.Connection:
        """Get database connection, raising if not initialized."""
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized."""
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
        """Store a context node.

        Args:
            node: The node to store (must have id, type, model_dump()).
            session_id: Session identifier for namespacing.
            metadata: Optional metadata. Auto-generated if not provided.

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

        # Create metadata if not provided
        if metadata is None:
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.WARM,
                size_bytes=len(node_data.encode("utf-8")),
                token_count=getattr(node, "token_count", 0) or 0,
                node_type=node.type.value,
                importance=getattr(getattr(node, "metadata", None), "importance", 0.5),
                tags=set(getattr(getattr(node, "metadata", None), "tags", [])),
            )

        now = datetime.now(UTC).isoformat()
        metadata_json = json.dumps(metadata.model_dump(mode="json"))
        tags_json = json.dumps(list(metadata.tags))

        await self._db.execute(
            """
            INSERT INTO context_nodes (
                node_id, session_id, version, node_data,
                tier, node_type, importance, size_bytes, token_count, tags,
                created_at, accessed_at, access_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id, session_id, version) DO UPDATE SET
                node_data = excluded.node_data,
                tier = excluded.tier,
                node_type = excluded.node_type,
                importance = excluded.importance,
                size_bytes = excluded.size_bytes,
                token_count = excluded.token_count,
                tags = excluded.tags,
                metadata_json = excluded.metadata_json
            """,
            (
                str(key.node_id),
                session_id,
                key.version,
                node_data,
                metadata.tier.value,
                metadata.node_type,
                metadata.importance,
                metadata.size_bytes,
                metadata.token_count,
                tags_json,
                now,
                now,
                0,
                metadata_json,
            ),
        )
        await self._db.commit()

        return key

    async def store_batch(
        self,
        nodes: list[Any],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes in a single transaction.

        Args:
            nodes: List of nodes to store.
            session_id: Session identifier for all nodes.

        Returns:
            List of StorageKeys in same order as input.
        """
        await self._ensure_initialized()

        keys = []
        # Use transaction for atomicity and performance
        async with self._db.execute("BEGIN IMMEDIATE"):
            for node in nodes:
                key = await self._store_single(node, session_id)
                keys.append(key)
        await self._db.commit()

        return keys

    async def _store_single(
        self,
        node: Any,
        session_id: str,
    ) -> StorageKey:
        """Store single node without commit (for batch use)."""
        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,
        )

        node_data = json.dumps(node.model_dump(mode="json"))
        now = datetime.now(UTC).isoformat()

        metadata = StorageMetadata(
            key=key,
            tier=StorageTier.WARM,
            size_bytes=len(node_data.encode("utf-8")),
            token_count=getattr(node, "token_count", 0) or 0,
            node_type=node.type.value,
            importance=getattr(getattr(node, "metadata", None), "importance", 0.5),
            tags=set(getattr(getattr(node, "metadata", None), "tags", [])),
        )

        metadata_json = json.dumps(metadata.model_dump(mode="json"))
        tags_json = json.dumps(list(metadata.tags))

        await self._db.execute(
            """
            INSERT INTO context_nodes (
                node_id, session_id, version, node_data,
                tier, node_type, importance, size_bytes, token_count, tags,
                created_at, accessed_at, access_count, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id, session_id, version) DO UPDATE SET
                node_data = excluded.node_data,
                tier = excluded.tier,
                importance = excluded.importance,
                metadata_json = excluded.metadata_json
            """,
            (
                str(key.node_id),
                session_id,
                key.version,
                node_data,
                metadata.tier.value,
                metadata.node_type,
                metadata.importance,
                metadata.size_bytes,
                metadata.token_count,
                tags_json,
                now,
                now,
                0,
                metadata_json,
            ),
        )

        return key

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Any | None:
        """Retrieve a node by key.

        Args:
            key: StorageKey from previous store operation.

        Returns:
            The node if found, None otherwise.
        """
        await self._ensure_initialized()

        cursor = await self._db.execute(
            """
            SELECT node_data FROM context_nodes
            WHERE node_id = ? AND session_id = ? AND version = ?
            """,
            (str(key.node_id), key.session_id, key.version),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        # Update access tracking
        await self._touch(key)
        await self._db.commit()

        node_data = json.loads(row[0])

        # Try to deserialize as ContextNode if available
        try:
            from context_core.graph import ContextNode

            return ContextNode.model_validate(node_data)
        except ImportError:
            # context-core not available
            return node_data
        except Exception:
            # Validation failed (e.g., mock data in tests)
            return node_data

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Any | None]:
        """Retrieve multiple nodes.

        Args:
            keys: List of StorageKeys.

        Returns:
            List of nodes (or None) in same order as keys.
        """
        await self._ensure_initialized()

        # Build query with placeholders
        if not keys:
            return []

        results: dict[str, Any] = {}

        # Query all at once for efficiency
        placeholders = ",".join(["(?, ?, ?)"] * len(keys))
        params = []
        for key in keys:
            params.extend([str(key.node_id), key.session_id, key.version])

        cursor = await self._db.execute(
            f"""
            SELECT node_id, session_id, version, node_data
            FROM context_nodes
            WHERE (node_id, session_id, version) IN ({placeholders})
            """,
            params,
        )

        rows = await cursor.fetchall()

        # Try to get ContextNode class
        context_node_cls = None
        try:
            from context_core.graph import ContextNode

            context_node_cls = ContextNode
        except ImportError:
            pass

        for row in rows:
            node_id, session_id, version, node_data = row
            key_str = f"{session_id}/{node_id}/{version}"
            data = json.loads(node_data)
            if context_node_cls:
                try:
                    results[key_str] = context_node_cls.model_validate(data)
                except Exception:
                    # Validation failed (e.g., mock data in tests)
                    results[key_str] = data
            else:
                results[key_str] = data

        # Update access tracking for found nodes
        for key in keys:
            key_str = f"{key.session_id}/{key.node_id}/{key.version}"
            if key_str in results:
                await self._touch(key)

        await self._db.commit()

        # Return in order
        return [results.get(f"{k.session_id}/{k.node_id}/{k.version}") for k in keys]

    async def _touch(self, key: StorageKey) -> None:
        """Update access tracking."""
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """
            UPDATE context_nodes
            SET accessed_at = ?, access_count = access_count + 1
            WHERE node_id = ? AND session_id = ? AND version = ?
            """,
            (now, str(key.node_id), key.session_id, key.version),
        )

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

        cursor = await self._db.execute(
            """
            DELETE FROM context_nodes
            WHERE node_id = ? AND session_id = ? AND version = ?
            """,
            (str(key.node_id), key.session_id, key.version),
        )
        await self._db.commit()

        return cursor.rowcount > 0

    async def exists(
        self,
        key: StorageKey,
    ) -> bool:
        """Check if key exists.

        Args:
            key: StorageKey to check.

        Returns:
            True if exists, False otherwise.
        """
        await self._ensure_initialized()

        cursor = await self._db.execute(
            """
            SELECT 1 FROM context_nodes
            WHERE node_id = ? AND session_id = ? AND version = ?
            LIMIT 1
            """,
            (str(key.node_id), key.session_id, key.version),
        )
        row = await cursor.fetchone()

        return row is not None

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> StorageMetadata | None:
        """Get metadata without retrieving node content.

        Args:
            key: StorageKey to get metadata for.

        Returns:
            StorageMetadata if found, None otherwise.
        """
        await self._ensure_initialized()

        cursor = await self._db.execute(
            """
            SELECT tier, node_type, importance, size_bytes, token_count,
                   tags, created_at, accessed_at, access_count
            FROM context_nodes
            WHERE node_id = ? AND session_id = ? AND version = ?
            """,
            (str(key.node_id), key.session_id, key.version),
        )
        row = await cursor.fetchone()

        if row is None:
            return None

        (
            tier,
            node_type,
            importance,
            size_bytes,
            token_count,
            tags_json,
            created_at,
            accessed_at,
            access_count,
        ) = row

        return StorageMetadata(
            key=key,
            tier=StorageTier(tier),
            node_type=node_type,
            importance=importance,
            size_bytes=size_bytes,
            token_count=token_count,
            tags=set(json.loads(tags_json)),
            created_at=datetime.fromisoformat(created_at),
            accessed_at=datetime.fromisoformat(accessed_at),
            access_count=access_count,
        )

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict[str, Any],
    ) -> bool:
        """Update metadata fields.

        Args:
            key: StorageKey of node to update.
            updates: Dictionary of fields to update.

        Returns:
            True if updated, False if not found.
        """
        await self._ensure_initialized()

        # Get current metadata
        metadata = await self.get_metadata(key)
        if metadata is None:
            return False

        # Apply updates
        metadata_dict = metadata.model_dump(mode="json")
        metadata_dict.update(updates)
        metadata_json = json.dumps(metadata_dict)

        # Update denormalized columns if present in updates
        set_clauses = ["metadata_json = ?"]
        params: list[Any] = [metadata_json]

        if "tier" in updates:
            set_clauses.append("tier = ?")
            params.append(updates["tier"])
        if "importance" in updates:
            set_clauses.append("importance = ?")
            params.append(updates["importance"])
        if "tags" in updates:
            set_clauses.append("tags = ?")
            tags = updates["tags"]
            params.append(json.dumps(list(tags) if isinstance(tags, set) else tags))

        params.extend([str(key.node_id), key.session_id, key.version])

        await self._db.execute(
            f"""
            UPDATE context_nodes
            SET {", ".join(set_clauses)}
            WHERE node_id = ? AND session_id = ? AND version = ?
            """,
            params,
        )
        await self._db.commit()

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
        """List keys for a session.

        Args:
            session_id: Session to list keys for.
            tier: Optional tier filter.
            node_type: Optional node type filter.
            limit: Maximum results.

        Returns:
            List of StorageKeys.
        """
        await self._ensure_initialized()

        query = (
            "SELECT node_id, session_id, version "
            "FROM context_nodes WHERE session_id = ?"
        )
        params: list[Any] = [session_id]

        if tier is not None:
            query += " AND tier = ?"
            params.append(tier.value)
        if node_type is not None:
            query += " AND node_type = ?"
            params.append(node_type)

        query += " LIMIT ?"
        params.append(limit)

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()

        return [
            StorageKey(
                session_id=row[1],
                node_id=UUID(row[0]),
                version=row[2],
            )
            for row in rows
        ]

    async def search_by_metadata(
        self,
        session_id: str,
        *,
        min_importance: float | None = None,
        tags: set[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        """Search by metadata criteria.

        Args:
            session_id: Session to search in.
            min_importance: Minimum importance threshold.
            tags: Tags to match (OR semantics).
            since: Only nodes created after this time.
            limit: Maximum results.

        Returns:
            List of (StorageKey, StorageMetadata) tuples, ordered by importance.
        """
        await self._ensure_initialized()

        query = """
            SELECT node_id, session_id, version, metadata_json, tags
            FROM context_nodes
            WHERE session_id = ?
        """
        params: list[Any] = [session_id]

        if min_importance is not None:
            query += " AND importance >= ?"
            params.append(min_importance)
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY importance DESC LIMIT ?"
        params.append(limit * 2 if tags else limit)  # Over-fetch if filtering by tags

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            node_id, sess_id, version, metadata_json, tags_json = row

            # Filter by tags if specified
            if tags:
                row_tags = set(json.loads(tags_json))
                if not (tags & row_tags):
                    continue

            key = StorageKey(
                session_id=sess_id,
                node_id=UUID(node_id),
                version=version,
            )
            metadata = StorageMetadata.model_validate(json.loads(metadata_json))
            results.append((key, metadata))

            if len(results) >= limit:
                break

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
            session_id: Optional session filter.

        Returns:
            StorageStats with aggregated data.
        """
        await self._ensure_initialized()

        where = "WHERE session_id = ?" if session_id else ""
        params: list[Any] = [session_id] if session_id else []

        # Aggregate query
        cursor = await self._db.execute(
            f"""
            SELECT
                COUNT(*) as total_items,
                COALESCE(SUM(size_bytes), 0) as total_size,
                COALESCE(SUM(token_count), 0) as total_tokens,
                COALESCE(AVG(access_count), 0) as avg_access,
                MIN(created_at) as oldest,
                MAX(created_at) as newest
            FROM context_nodes
            {where}
            """,
            params,
        )
        row = await cursor.fetchone()

        if row is None:
            # No data found
            return StorageStats(
                total_items=0,
                total_size_bytes=0,
                total_tokens=0,
                items_by_tier={},
                size_by_tier={},
                avg_access_count=0.0,
                oldest_item=None,
                newest_item=None,
            )

        total_items, total_size, total_tokens, avg_access, oldest, newest = row

        # Tier breakdown
        cursor = await self._db.execute(
            f"""
            SELECT tier, COUNT(*), SUM(size_bytes)
            FROM context_nodes
            {where}
            GROUP BY tier
            """,
            params,
        )
        tier_rows = await cursor.fetchall()

        items_by_tier = {}
        size_by_tier = {}
        for tier_row in tier_rows:
            tier_name, count, size = tier_row
            items_by_tier[tier_name] = count
            size_by_tier[tier_name] = size or 0

        return StorageStats(
            total_items=total_items or 0,
            total_size_bytes=total_size or 0,
            total_tokens=total_tokens or 0,
            items_by_tier=items_by_tier,
            size_by_tier=size_by_tier,
            avg_access_count=float(avg_access or 0),
            oldest_item=datetime.fromisoformat(oldest) if oldest else None,
            newest_item=datetime.fromisoformat(newest) if newest else None,
        )

    async def close(self) -> None:
        """Close database connection.

        Safe to call multiple times.
        """
        if self._conn is not None:
            await self._db.close()
            self._conn = None
            self._initialized = False
