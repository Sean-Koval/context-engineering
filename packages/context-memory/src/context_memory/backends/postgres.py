"""PostgresStore - PostgreSQL storage backend with connection pooling.

This module provides a production-grade storage backend using PostgreSQL
with asyncpg for high-performance async operations. Suitable for deployments
requiring concurrent access, complex queries, and scalability.

Features:
    - Connection pooling for concurrent access
    - JSONB for efficient node and metadata storage
    - Denormalized columns for fast filtering
    - Composite primary key with version support
    - Automatic schema creation
    - Access tracking with atomic updates

Schema:
    CREATE TABLE context_nodes (
        node_id UUID NOT NULL,
        session_id TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        node_data JSONB NOT NULL,
        tier TEXT NOT NULL DEFAULT 'warm',
        node_type TEXT NOT NULL,
        importance REAL NOT NULL DEFAULT 0.5,
        size_bytes INTEGER NOT NULL,
        token_count INTEGER NOT NULL,
        tags JSONB NOT NULL DEFAULT '[]',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        access_count INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (node_id, session_id, version)
    );
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
    import asyncpg


# Schema DDL - executed on first connection
_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_nodes (
    -- Composite primary key for versioning
    node_id UUID NOT NULL,
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,

    -- Node data (JSONB for efficient storage and querying)
    node_data JSONB NOT NULL,

    -- Denormalized metadata for fast queries
    tier TEXT NOT NULL DEFAULT 'warm',
    node_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    size_bytes INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- Timestamps (TIMESTAMPTZ for timezone awareness)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INTEGER NOT NULL DEFAULT 0,

    PRIMARY KEY (node_id, session_id, version)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_context_session
    ON context_nodes(session_id);
CREATE INDEX IF NOT EXISTS idx_context_session_tier
    ON context_nodes(session_id, tier);
CREATE INDEX IF NOT EXISTS idx_context_session_type
    ON context_nodes(session_id, node_type);
CREATE INDEX IF NOT EXISTS idx_context_importance
    ON context_nodes(importance DESC);
CREATE INDEX IF NOT EXISTS idx_context_created
    ON context_nodes(created_at DESC);
"""


class PostgresStore:
    """PostgreSQL storage backend with connection pooling.

    This backend uses asyncpg for high-performance async PostgreSQL operations.
    It supports all MemoryStore operations including batch operations, metadata
    queries, and access tracking.

    Attributes:
        connection_string: PostgreSQL connection URL.
        pool_size: Maximum connections in the pool.

    Example:
        >>> store = PostgresStore("postgresql://user:pass@localhost/db")
        >>> await store.initialize()
        >>> key = await store.store(node, "session-123")
        >>> retrieved = await store.retrieve(key)
        >>> await store.close()

    Note:
        Requires the 'postgres' optional dependency:
        pip install context-memory[postgres]
    """

    def __init__(
        self,
        connection_string: str,
        *,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        """Initialize the PostgreSQL store.

        Args:
            connection_string: PostgreSQL connection URL
                (e.g., "postgresql://user:pass@localhost:5432/dbname").
            pool_min_size: Minimum connections to maintain in pool.
            pool_max_size: Maximum connections in pool.
        """
        self._connection_string = connection_string
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: asyncpg.Pool | None = None
        self._initialized = False

    @property
    def _db(self) -> asyncpg.Pool:
        """Get connection pool, raising if not initialized."""
        if self._pool is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._pool

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """Initialize connection pool and create schema.

        Creates the connection pool and ensures the database schema exists.
        Safe to call multiple times (idempotent).
        """
        if self._initialized:
            return

        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg is required for PostgresStore. "
                "Install with: pip install context-memory[postgres]"
            ) from e

        self._pool = await asyncpg.create_pool(
            self._connection_string,
            min_size=self._pool_min_size,
            max_size=self._pool_max_size,
        )

        # Create schema
        async with self._db.acquire() as conn:
            await conn.execute(_SCHEMA)

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
        """Store a context node.

        Args:
            node: The ContextNode to store (or compatible mock object).
            session_id: Session identifier for namespacing.
            metadata: Optional metadata. If not provided, auto-generated.

        Returns:
            StorageKey for retrieval.
        """
        await self._ensure_initialized()

        # Create storage key
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
                token_count=node.token_count or 0,
                node_type=node.type.value,
                importance=node.metadata.importance
                if hasattr(node.metadata, "importance")
                else 0.5,
                tags=set(node.metadata.tags)
                if hasattr(node.metadata, "tags")
                else set(),
            )

        now = datetime.now(UTC)
        tags_json = json.dumps(list(metadata.tags))

        async with self._db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO context_nodes (
                    node_id, session_id, version, node_data,
                    tier, node_type, importance, size_bytes, token_count, tags,
                    created_at, accessed_at, access_count
                ) VALUES (
                    $1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10::jsonb,
                    $11, $12, $13
                )
                ON CONFLICT (node_id, session_id, version) DO UPDATE SET
                    node_data = EXCLUDED.node_data,
                    tier = EXCLUDED.tier,
                    node_type = EXCLUDED.node_type,
                    importance = EXCLUDED.importance,
                    size_bytes = EXCLUDED.size_bytes,
                    token_count = EXCLUDED.token_count,
                    tags = EXCLUDED.tags,
                    accessed_at = EXCLUDED.accessed_at
                """,
                key.node_id,
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
            )

        return key

    async def store_batch(
        self,
        nodes: list[Any],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes in a transaction.

        Args:
            nodes: List of ContextNodes to store.
            session_id: Session identifier for all nodes.

        Returns:
            List of StorageKeys in the same order as input nodes.
        """
        await self._ensure_initialized()

        keys = []
        async with self._db.acquire() as conn, conn.transaction():
            for node in nodes:
                key = await self._store_single(conn, node, session_id)
                keys.append(key)

        return keys

    async def _store_single(
        self,
        conn: Any,  # asyncpg.Connection
        node: Any,
        session_id: str,
    ) -> StorageKey:
        """Store a single node using an existing connection."""
        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,
        )

        node_data = json.dumps(node.model_dump(mode="json"))
        now = datetime.now(UTC)

        token_count = node.token_count or 0
        importance = (
            node.metadata.importance if hasattr(node.metadata, "importance") else 0.5
        )
        tags = list(node.metadata.tags) if hasattr(node.metadata, "tags") else []

        await conn.execute(
            """
            INSERT INTO context_nodes (
                node_id, session_id, version, node_data,
                tier, node_type, importance, size_bytes, token_count, tags,
                created_at, accessed_at, access_count
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $10::jsonb,
                $11, $12, $13
            )
            ON CONFLICT (node_id, session_id, version) DO UPDATE SET
                node_data = EXCLUDED.node_data,
                accessed_at = EXCLUDED.accessed_at
            """,
            key.node_id,
            session_id,
            key.version,
            node_data,
            StorageTier.WARM.value,
            node.type.value,
            importance,
            len(node_data.encode("utf-8")),
            token_count,
            json.dumps(tags),
            now,
            now,
            0,
        )

        return key

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Any | None:
        """Retrieve a node by key.

        Also updates access tracking (accessed_at, access_count).

        Args:
            key: StorageKey from previous store operation.

        Returns:
            The node if found, None otherwise.
        """
        await self._ensure_initialized()

        async with self._db.acquire() as conn:
            # Update access tracking and return data in one query
            row = await conn.fetchrow(
                """
                UPDATE context_nodes
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE node_id = $1 AND session_id = $2 AND version = $3
                RETURNING node_data
                """,
                key.node_id,
                key.session_id,
                key.version,
            )

            if row is None:
                return None

            node_data = json.loads(row["node_data"])

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
        """Retrieve multiple nodes by their keys.

        Args:
            keys: List of StorageKeys to retrieve.

        Returns:
            List of nodes (or None for missing keys) in the same order.
        """
        await self._ensure_initialized()

        if not keys:
            return []

        results: dict[str, Any] = {}

        async with self._db.acquire() as conn:
            # Build arrays for batch query
            node_ids = [key.node_id for key in keys]
            session_ids = list({key.session_id for key in keys})

            # Update access tracking and fetch all matching rows
            rows = await conn.fetch(
                """
                UPDATE context_nodes
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE node_id = ANY($1) AND session_id = ANY($2)
                RETURNING node_id, session_id, version, node_data
                """,
                node_ids,
                session_ids,
            )

            # Try to get ContextNode class
            context_node_cls = None
            try:
                from context_core.graph import ContextNode

                context_node_cls = ContextNode
            except ImportError:
                pass

            for row in rows:
                key_str = f"{row['session_id']}/{row['node_id']}/{row['version']}"
                data = json.loads(row["node_data"])
                if context_node_cls:
                    try:
                        results[key_str] = context_node_cls.model_validate(data)
                    except Exception:
                        results[key_str] = data
                else:
                    results[key_str] = data

        # Return in order
        return [results.get(f"{k.session_id}/{k.node_id}/{k.version}") for k in keys]

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

        async with self._db.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM context_nodes
                WHERE node_id = $1 AND session_id = $2 AND version = $3
                """,
                key.node_id,
                key.session_id,
                key.version,
            )

            # asyncpg returns "DELETE N" where N is rows affected
            return result == "DELETE 1"

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

        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1 FROM context_nodes
                WHERE node_id = $1 AND session_id = $2 AND version = $3
                """,
                key.node_id,
                key.session_id,
                key.version,
            )

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

        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT tier, node_type, importance, size_bytes, token_count,
                       tags, created_at, accessed_at, access_count
                FROM context_nodes
                WHERE node_id = $1 AND session_id = $2 AND version = $3
                """,
                key.node_id,
                key.session_id,
                key.version,
            )

            if row is None:
                return None

            return StorageMetadata(
                key=key,
                tier=StorageTier(row["tier"]),
                node_type=row["node_type"],
                importance=row["importance"],
                size_bytes=row["size_bytes"],
                token_count=row["token_count"],
                tags=set(row["tags"]),  # JSONB array -> set
                created_at=row["created_at"],
                accessed_at=row["accessed_at"],
                access_count=row["access_count"],
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

        if not updates:
            return True

        # Map field names to columns
        column_map = {
            "tier": "tier",
            "importance": "importance",
            "tags": "tags",
            "accessed_at": "accessed_at",
            "access_count": "access_count",
        }

        set_clauses = []
        params: list[Any] = []
        param_idx = 4  # First 3 are for WHERE clause

        for field, value in updates.items():
            if field in column_map:
                col = column_map[field]
                if field == "tier" and isinstance(value, StorageTier):
                    value = value.value
                elif field == "tags" and isinstance(value, set):
                    value = json.dumps(list(value))
                    set_clauses.append(f"{col} = ${param_idx}::jsonb")
                else:
                    set_clauses.append(f"{col} = ${param_idx}")
                params.append(value)
                param_idx += 1

        if not set_clauses:
            return True

        query = f"""
            UPDATE context_nodes
            SET {", ".join(set_clauses)}
            WHERE node_id = $1 AND session_id = $2 AND version = $3
        """

        async with self._db.acquire() as conn:
            result = await conn.execute(
                query,
                key.node_id,
                key.session_id,
                key.version,
                *params,
            )

            return result == "UPDATE 1"

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
            tier: If provided, filter by storage tier.
            node_type: If provided, filter by node type.
            limit: Maximum number of keys to return.

        Returns:
            List of StorageKeys matching the criteria.
        """
        await self._ensure_initialized()

        query = """
            SELECT node_id, version FROM context_nodes
            WHERE session_id = $1
        """
        params: list[Any] = [session_id]
        param_idx = 2

        if tier is not None:
            query += f" AND tier = ${param_idx}"
            params.append(tier.value)
            param_idx += 1

        if node_type is not None:
            query += f" AND node_type = ${param_idx}"
            params.append(node_type)
            param_idx += 1

        query += f" LIMIT ${param_idx}"
        params.append(limit)

        async with self._db.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [
                StorageKey(
                    session_id=session_id,
                    node_id=row["node_id"],
                    version=row["version"],
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
        """Search for nodes by metadata criteria.

        Args:
            session_id: Session to search in.
            min_importance: Minimum importance threshold.
            tags: Tags to match (OR semantics).
            since: Only return nodes created after this time.
            limit: Maximum number of results.

        Returns:
            List of (StorageKey, StorageMetadata) tuples, ordered by importance.
        """
        await self._ensure_initialized()

        query = """
            SELECT node_id, version, tier, node_type, importance,
                   size_bytes, token_count, tags,
                   created_at, accessed_at, access_count
            FROM context_nodes
            WHERE session_id = $1
        """
        params: list[Any] = [session_id]
        param_idx = 2

        if min_importance is not None:
            query += f" AND importance >= ${param_idx}"
            params.append(min_importance)
            param_idx += 1

        if since is not None:
            query += f" AND created_at >= ${param_idx}"
            params.append(since)
            param_idx += 1

        query += f" ORDER BY importance DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._db.acquire() as conn:
            rows = await conn.fetch(query, *params)

            results: list[tuple[StorageKey, StorageMetadata]] = []

            for row in rows:
                key = StorageKey(
                    session_id=session_id,
                    node_id=row["node_id"],
                    version=row["version"],
                )

                row_tags = set(row["tags"])

                # Filter by tags if specified (OR semantics)
                if tags and not (tags & row_tags):
                    continue

                metadata = StorageMetadata(
                    key=key,
                    tier=StorageTier(row["tier"]),
                    node_type=row["node_type"],
                    importance=row["importance"],
                    size_bytes=row["size_bytes"],
                    token_count=row["token_count"],
                    tags=row_tags,
                    created_at=row["created_at"],
                    accessed_at=row["accessed_at"],
                    access_count=row["access_count"],
                )

                results.append((key, metadata))

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
            session_id: If provided, stats for this session only.
                If None, aggregate stats across all sessions.

        Returns:
            StorageStats with counts, sizes, and tier breakdowns.
        """
        await self._ensure_initialized()

        if session_id:
            where = "WHERE session_id = $1"
            params: list[Any] = [session_id]
        else:
            where = ""
            params = []

        async with self._db.acquire() as conn:
            row = await conn.fetchrow(
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
                *params,
            )

            if row is None or row["total_items"] == 0:
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

            tier_rows = await conn.fetch(
                f"""
                SELECT tier, COUNT(*) as cnt, COALESCE(SUM(size_bytes), 0) as size
                FROM context_nodes
                {where}
                GROUP BY tier
                """,
                *params,
            )

            items_by_tier = {r["tier"]: r["cnt"] for r in tier_rows}
            size_by_tier = {r["tier"]: r["size"] for r in tier_rows}

            return StorageStats(
                total_items=row["total_items"],
                total_size_bytes=row["total_size"],
                total_tokens=row["total_tokens"],
                items_by_tier=items_by_tier,
                size_by_tier=size_by_tier,
                avg_access_count=float(row["avg_access"]),
                oldest_item=row["oldest"],
                newest_item=row["newest"],
            )

    async def close(self) -> None:
        """Close the connection pool.

        Safe to call multiple times.
        """
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False
