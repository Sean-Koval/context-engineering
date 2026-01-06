# Context-Memory: Detailed Implementation Specification

> **Phase 3 Memory & Storage** | Estimated: 4-5 weeks | Priority: P0
> This document provides implementable specifications for persistent memory and retrieval.

---

## Table of Contents
1. [Package Overview](#package-overview)
2. [Storage Architecture](#storage-architecture)
3. [Component 1: MemoryStore Protocol](#component-1-memorystore-protocol)
4. [Component 2: Storage Backends](#component-2-storage-backends)
5. [Component 3: TieredStorage](#component-3-tieredstorage)
6. [Component 4: MemoryRetriever](#component-4-memoryretriever)
7. [Component 5: ArtifactManager](#component-5-artifactmanager)
8. [Component 6: WorkingMemory](#component-6-workingmemory)
9. [Integration Patterns](#integration-patterns)
10. [Task Breakdown](#task-breakdown)
11. [Test Specifications](#test-specifications)

---

## Package Overview

### Purpose
`context-memory` provides persistent storage, tiered caching, and intelligent retrieval for context that exceeds working memory limits.

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MEMORY ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    WorkingMemory (Cache)                         │   │
│  │    Fast access, LRU eviction, sync with backing store           │   │
│  │    Capacity: ~50% of token budget                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      TieredStorage                               │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  HOT (Redis/Memory)    │  < 1 hour old, high access             │   │
│  │  WARM (PostgreSQL)     │  < 24 hours, moderate access           │   │
│  │  COLD (S3/Filesystem)  │  > 24 hours, archived                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                        │
│                                 ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     MemoryRetriever                              │   │
│  │    Semantic, Entity, Temporal, Task-Pattern strategies          │   │
│  │    Ensemble ranking, relevance scoring                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Dependencies

```toml
[project]
name = "context-memory"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "context-core>=0.1.0",
    "pydantic>=2.0",
    "aiofiles>=23.0",
]

[project.optional-dependencies]
postgres = ["asyncpg>=0.29", "pgvector>=0.2"]
redis = ["redis>=5.0"]
s3 = ["aioboto3>=12.0"]
all = ["context-memory[postgres,redis,s3]"]
```

### Module Structure

```
context_memory/
├── __init__.py
├── py.typed
├── types.py                  # Shared types, StorageKey, StorageMetadata
├── store.py                  # MemoryStore protocol
├── backends/
│   ├── __init__.py
│   ├── base.py               # StorageBackend protocol
│   ├── filesystem.py         # FileSystemStore
│   ├── postgres.py           # PostgresStore
│   ├── redis.py              # RedisStore
│   └── s3.py                 # S3Store
├── tiered.py                 # TieredStorage
├── working.py                # WorkingMemory
├── retrieval/
│   ├── __init__.py
│   ├── base.py               # RetrievalStrategy protocol
│   ├── semantic.py           # SemanticRetrieval
│   ├── entity.py             # EntityRetrieval
│   ├── temporal.py           # TemporalRetrieval
│   ├── task.py               # TaskPatternRetrieval
│   └── ensemble.py           # EnsembleRetriever
└── artifacts/
    ├── __init__.py
    ├── manager.py            # ArtifactManager
    └── versioning.py         # VersionedArtifact
```

---

## Storage Architecture

### Storage Tier Characteristics

| Tier | Backend | Latency | Capacity | Access Pattern | Retention |
|------|---------|---------|----------|----------------|-----------|
| **Hot** | Redis/Memory | < 10ms | ~10% | Frequent | < 1 hour |
| **Warm** | PostgreSQL | < 50ms | ~30% | Moderate | < 24 hours |
| **Cold** | S3/Filesystem | < 500ms | Unlimited | Rare | Indefinite |

### Automatic Tier Migration

```
┌──────────────────────────────────────────────────────────────────────┐
│                      TIER MIGRATION FLOW                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [New Context]                                                        │
│       │                                                               │
│       ▼                                                               │
│  ┌─────────┐                                                          │
│  │   HOT   │ ◄─── Access resets age                                  │
│  └────┬────┘                                                          │
│       │                                                               │
│       │ Age > 1 hour OR eviction pressure                            │
│       ▼                                                               │
│  ┌─────────┐                                                          │
│  │  WARM   │ ◄─── Access promotes to HOT                             │
│  └────┬────┘                                                          │
│       │                                                               │
│       │ Age > 24 hours OR low importance                             │
│       ▼                                                               │
│  ┌─────────┐                                                          │
│  │  COLD   │ ◄─── Access promotes to WARM                            │
│  └─────────┘                                                          │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: MemoryStore Protocol

### 1.1 Type Definitions

```python
# context_memory/types.py
from enum import Enum
from typing import Any, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

class StorageTier(str, Enum):
    """Storage tier for tiered storage."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class StorageKey(BaseModel):
    """Unique key for stored items."""
    session_id: str
    node_id: UUID
    version: int = 1

    def __str__(self) -> str:
        return f"{self.session_id}/{self.node_id}/{self.version}"

    @classmethod
    def from_string(cls, s: str) -> "StorageKey":
        parts = s.split("/")
        return cls(
            session_id=parts[0],
            node_id=UUID(parts[1]),
            version=int(parts[2]) if len(parts) > 2 else 1,
        )


class StorageMetadata(BaseModel):
    """Metadata for stored items."""
    key: StorageKey
    tier: StorageTier = StorageTier.HOT
    size_bytes: int
    token_count: int

    # Temporal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0

    # Classification
    node_type: str
    importance: float = 0.5
    tags: set[str] = Field(default_factory=set)

    # Compression state
    is_compressed: bool = False
    original_size_bytes: Optional[int] = None

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1


class StorageStats(BaseModel):
    """Statistics for a storage backend."""
    total_items: int
    total_size_bytes: int
    total_tokens: int
    items_by_tier: dict[str, int] = Field(default_factory=dict)
    size_by_tier: dict[str, int] = Field(default_factory=dict)
    avg_access_count: float = 0.0
    oldest_item: Optional[datetime] = None
    newest_item: Optional[datetime] = None


class RetrievalResult(BaseModel):
    """Result from memory retrieval."""
    node: "ContextNode"  # Forward reference
    score: float
    source_tier: StorageTier
    retrieval_method: str
    latency_ms: float
```

### 1.2 MemoryStore Protocol

```python
# context_memory/store.py
from typing import Protocol, Optional, AsyncIterator, runtime_checkable
from uuid import UUID

from context_core.graph import ContextNode, ContextGraph

from .types import StorageKey, StorageMetadata, StorageStats, StorageTier


@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for context memory storage.

    All methods are async to support various backends.
    Implementations should handle their own connection pooling.
    """

    async def store(
        self,
        node: ContextNode,
        session_id: str,
        metadata: Optional[StorageMetadata] = None,
    ) -> StorageKey:
        """
        Store a context node.

        Args:
            node: The node to store
            session_id: Session identifier for namespacing
            metadata: Optional metadata (auto-generated if not provided)

        Returns:
            Storage key for retrieval
        """
        ...

    async def store_batch(
        self,
        nodes: list[ContextNode],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes efficiently."""
        ...

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Optional[ContextNode]:
        """
        Retrieve a node by key.

        Returns None if not found.
        """
        ...

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Optional[ContextNode]]:
        """Retrieve multiple nodes by keys."""
        ...

    async def delete(self, key: StorageKey) -> bool:
        """Delete a node. Returns True if existed."""
        ...

    async def exists(self, key: StorageKey) -> bool:
        """Check if a key exists."""
        ...

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> Optional[StorageMetadata]:
        """Get metadata without retrieving content."""
        ...

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict,
    ) -> bool:
        """Update metadata fields."""
        ...

    async def list_keys(
        self,
        session_id: str,
        *,
        tier: Optional[StorageTier] = None,
        node_type: Optional[str] = None,
        limit: int = 1000,
    ) -> list[StorageKey]:
        """List keys matching criteria."""
        ...

    async def search_by_metadata(
        self,
        session_id: str,
        *,
        min_importance: Optional[float] = None,
        tags: Optional[set[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        """Search by metadata fields."""
        ...

    async def stats(self, session_id: Optional[str] = None) -> StorageStats:
        """Get storage statistics."""
        ...

    async def close(self) -> None:
        """Close connections and cleanup."""
        ...
```

---

## Component 2: Storage Backends

### 2.1 FileSystemStore

```python
# context_memory/backends/filesystem.py
import os
import json
import aiofiles
import aiofiles.os
from pathlib import Path
from typing import Optional
from datetime import datetime
from uuid import UUID
import hashlib

from context_core.graph import ContextNode

from ..types import StorageKey, StorageMetadata, StorageStats, StorageTier
from ..store import MemoryStore


class FileSystemStore:
    """
    File-based storage backend.

    Directory structure:
    {base_path}/
    ├── {session_id}/
    │   ├── nodes/
    │   │   ├── {node_id}.json
    │   │   └── ...
    │   ├── metadata/
    │   │   ├── {node_id}.meta.json
    │   │   └── ...
    │   └── index.json
    └── ...
    """

    def __init__(
        self,
        base_path: str | Path,
        create_if_missing: bool = True,
    ):
        self._base_path = Path(base_path)
        if create_if_missing:
            self._base_path.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get path for a session."""
        # Sanitize session_id for filesystem
        safe_id = hashlib.md5(session_id.encode()).hexdigest()[:16]
        return self._base_path / safe_id

    def _node_path(self, key: StorageKey) -> Path:
        """Get path for a node file."""
        session_path = self._session_path(key.session_id)
        return session_path / "nodes" / f"{key.node_id}.v{key.version}.json"

    def _metadata_path(self, key: StorageKey) -> Path:
        """Get path for metadata file."""
        session_path = self._session_path(key.session_id)
        return session_path / "metadata" / f"{key.node_id}.v{key.version}.meta.json"

    async def store(
        self,
        node: ContextNode,
        session_id: str,
        metadata: Optional[StorageMetadata] = None,
    ) -> StorageKey:
        """Store a node to filesystem."""
        # Create key
        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,  # Could implement versioning
        )

        # Ensure directories exist
        node_path = self._node_path(key)
        metadata_path = self._metadata_path(key)
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
                size_bytes=len(node_json.encode()),
                token_count=node.token_count or 0,
                node_type=node.type.value,
                importance=node.metadata.importance,
                tags=node.metadata.tags,
            )

        # Write files
        async with aiofiles.open(node_path, 'w') as f:
            await f.write(node_json)

        metadata_json = json.dumps(metadata.model_dump(mode="json"), indent=2)
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(metadata_json)

        return key

    async def store_batch(
        self,
        nodes: list[ContextNode],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes."""
        keys = []
        for node in nodes:
            key = await self.store(node, session_id)
            keys.append(key)
        return keys

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Optional[ContextNode]:
        """Retrieve a node from filesystem."""
        node_path = self._node_path(key)

        if not node_path.exists():
            return None

        async with aiofiles.open(node_path, 'r') as f:
            node_json = await f.read()

        node_data = json.loads(node_json)
        node = ContextNode.model_validate(node_data)

        # Update access metadata
        await self._touch_metadata(key)

        return node

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Optional[ContextNode]]:
        """Retrieve multiple nodes."""
        return [await self.retrieve(key) for key in keys]

    async def delete(self, key: StorageKey) -> bool:
        """Delete a node and its metadata."""
        node_path = self._node_path(key)
        metadata_path = self._metadata_path(key)

        existed = node_path.exists()

        if node_path.exists():
            await aiofiles.os.remove(node_path)
        if metadata_path.exists():
            await aiofiles.os.remove(metadata_path)

        return existed

    async def exists(self, key: StorageKey) -> bool:
        """Check if key exists."""
        return self._node_path(key).exists()

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> Optional[StorageMetadata]:
        """Get metadata for a key."""
        metadata_path = self._metadata_path(key)

        if not metadata_path.exists():
            return None

        async with aiofiles.open(metadata_path, 'r') as f:
            metadata_json = await f.read()

        return StorageMetadata.model_validate(json.loads(metadata_json))

    async def _touch_metadata(self, key: StorageKey) -> None:
        """Update access time in metadata."""
        metadata = await self.get_metadata(key)
        if metadata:
            metadata.touch()
            await self.update_metadata(key, {
                "accessed_at": metadata.accessed_at.isoformat(),
                "access_count": metadata.access_count,
            })

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict,
    ) -> bool:
        """Update metadata fields."""
        metadata = await self.get_metadata(key)
        if not metadata:
            return False

        metadata_dict = metadata.model_dump(mode="json")
        metadata_dict.update(updates)

        metadata_path = self._metadata_path(key)
        async with aiofiles.open(metadata_path, 'w') as f:
            await f.write(json.dumps(metadata_dict, indent=2))

        return True

    async def list_keys(
        self,
        session_id: str,
        *,
        tier: Optional[StorageTier] = None,
        node_type: Optional[str] = None,
        limit: int = 1000,
    ) -> list[StorageKey]:
        """List keys for a session."""
        session_path = self._session_path(session_id)
        nodes_path = session_path / "nodes"

        if not nodes_path.exists():
            return []

        keys = []
        for file_path in nodes_path.glob("*.json"):
            # Parse filename: {node_id}.v{version}.json
            name = file_path.stem
            parts = name.rsplit('.v', 1)
            node_id = UUID(parts[0])
            version = int(parts[1]) if len(parts) > 1 else 1

            key = StorageKey(
                session_id=session_id,
                node_id=node_id,
                version=version,
            )

            # Apply filters
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
        min_importance: Optional[float] = None,
        tags: Optional[set[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        """Search by metadata criteria."""
        all_keys = await self.list_keys(session_id, limit=10000)
        results = []

        for key in all_keys:
            metadata = await self.get_metadata(key)
            if not metadata:
                continue

            # Apply filters
            if min_importance and metadata.importance < min_importance:
                continue
            if tags and not (tags & metadata.tags):
                continue
            if since and metadata.created_at < since:
                continue

            results.append((key, metadata))

            if len(results) >= limit:
                break

        return results

    async def stats(self, session_id: Optional[str] = None) -> StorageStats:
        """Get storage statistics."""
        total_items = 0
        total_size = 0
        total_tokens = 0
        items_by_tier: dict[str, int] = {}
        size_by_tier: dict[str, int] = {}
        access_counts = []
        oldest: Optional[datetime] = None
        newest: Optional[datetime] = None

        sessions = [session_id] if session_id else [
            d.name for d in self._base_path.iterdir() if d.is_dir()
        ]

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
            avg_access_count=sum(access_counts) / len(access_counts) if access_counts else 0,
            oldest_item=oldest,
            newest_item=newest,
        )

    async def close(self) -> None:
        """No cleanup needed for filesystem."""
        pass
```

### 2.2 PostgresStore

```python
# context_memory/backends/postgres.py
from typing import Optional
from datetime import datetime
from uuid import UUID
import json

from context_core.graph import ContextNode

from ..types import StorageKey, StorageMetadata, StorageStats, StorageTier
from ..store import MemoryStore


class PostgresStore:
    """
    PostgreSQL storage backend with pgvector support.

    Schema:
    CREATE TABLE context_nodes (
        id UUID PRIMARY KEY,
        session_id TEXT NOT NULL,
        version INT DEFAULT 1,
        node_data JSONB NOT NULL,
        metadata JSONB NOT NULL,
        embedding vector(384),
        created_at TIMESTAMPTZ DEFAULT NOW(),
        accessed_at TIMESTAMPTZ DEFAULT NOW(),
        access_count INT DEFAULT 0
    );

    CREATE INDEX idx_session ON context_nodes(session_id);
    CREATE INDEX idx_session_type ON context_nodes(session_id, (metadata->>'node_type'));
    CREATE INDEX idx_importance ON context_nodes((metadata->>'importance')::float);
    CREATE INDEX idx_embedding ON context_nodes USING ivfflat (embedding vector_cosine_ops);
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
    ):
        self._connection_string = connection_string
        self._pool_size = pool_size
        self._pool = None

    async def _ensure_pool(self):
        """Ensure connection pool is initialized."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self._connection_string,
                min_size=2,
                max_size=self._pool_size,
            )

    async def _ensure_schema(self):
        """Ensure database schema exists."""
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS context_nodes (
                    id UUID PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    version INT DEFAULT 1,
                    node_data JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    tier TEXT DEFAULT 'warm',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    accessed_at TIMESTAMPTZ DEFAULT NOW(),
                    access_count INT DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_context_session
                    ON context_nodes(session_id);
                CREATE INDEX IF NOT EXISTS idx_context_tier
                    ON context_nodes(session_id, tier);
                CREATE INDEX IF NOT EXISTS idx_context_importance
                    ON context_nodes((metadata->>'importance')::float DESC);
                CREATE INDEX IF NOT EXISTS idx_context_created
                    ON context_nodes(created_at DESC);
            """)

    async def store(
        self,
        node: ContextNode,
        session_id: str,
        metadata: Optional[StorageMetadata] = None,
    ) -> StorageKey:
        """Store node in PostgreSQL."""
        await self._ensure_pool()

        key = StorageKey(
            session_id=session_id,
            node_id=node.id,
            version=1,
        )

        node_data = json.dumps(node.model_dump(mode="json"))

        if metadata is None:
            metadata = StorageMetadata(
                key=key,
                tier=StorageTier.WARM,
                size_bytes=len(node_data.encode()),
                token_count=node.token_count or 0,
                node_type=node.type.value,
                importance=node.metadata.importance,
                tags=node.metadata.tags,
            )

        metadata_json = json.dumps(metadata.model_dump(mode="json"))

        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO context_nodes (id, session_id, version, node_data, metadata, tier)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)
                ON CONFLICT (id) DO UPDATE SET
                    node_data = EXCLUDED.node_data,
                    metadata = EXCLUDED.metadata,
                    tier = EXCLUDED.tier
            """, node.id, session_id, key.version, node_data, metadata_json, metadata.tier.value)

        return key

    async def store_batch(
        self,
        nodes: list[ContextNode],
        session_id: str,
    ) -> list[StorageKey]:
        """Store multiple nodes efficiently."""
        await self._ensure_pool()

        keys = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for node in nodes:
                    key = await self.store(node, session_id)
                    keys.append(key)

        return keys

    async def retrieve(
        self,
        key: StorageKey,
    ) -> Optional[ContextNode]:
        """Retrieve node from PostgreSQL."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE context_nodes
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE id = $1 AND session_id = $2
                RETURNING node_data
            """, key.node_id, key.session_id)

            if row is None:
                return None

            node_data = json.loads(row['node_data'])
            return ContextNode.model_validate(node_data)

    async def retrieve_batch(
        self,
        keys: list[StorageKey],
    ) -> list[Optional[ContextNode]]:
        """Retrieve multiple nodes."""
        await self._ensure_pool()

        results: dict[UUID, ContextNode] = {}

        async with self._pool.acquire() as conn:
            node_ids = [k.node_id for k in keys]
            session_ids = [k.session_id for k in keys]

            rows = await conn.fetch("""
                UPDATE context_nodes
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE id = ANY($1) AND session_id = ANY($2)
                RETURNING id, node_data
            """, node_ids, session_ids)

            for row in rows:
                node_data = json.loads(row['node_data'])
                node = ContextNode.model_validate(node_data)
                results[row['id']] = node

        return [results.get(k.node_id) for k in keys]

    async def delete(self, key: StorageKey) -> bool:
        """Delete node from PostgreSQL."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM context_nodes
                WHERE id = $1 AND session_id = $2
            """, key.node_id, key.session_id)

            return result == "DELETE 1"

    async def exists(self, key: StorageKey) -> bool:
        """Check if key exists."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT 1 FROM context_nodes
                WHERE id = $1 AND session_id = $2
            """, key.node_id, key.session_id)

            return row is not None

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> Optional[StorageMetadata]:
        """Get metadata for key."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT metadata, tier, created_at, accessed_at, access_count
                FROM context_nodes
                WHERE id = $1 AND session_id = $2
            """, key.node_id, key.session_id)

            if row is None:
                return None

            metadata_dict = json.loads(row['metadata'])
            metadata_dict['tier'] = row['tier']
            metadata_dict['created_at'] = row['created_at'].isoformat()
            metadata_dict['accessed_at'] = row['accessed_at'].isoformat()
            metadata_dict['access_count'] = row['access_count']

            return StorageMetadata.model_validate(metadata_dict)

    async def update_metadata(
        self,
        key: StorageKey,
        updates: dict,
    ) -> bool:
        """Update metadata fields."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            # Get current metadata
            row = await conn.fetchrow("""
                SELECT metadata FROM context_nodes
                WHERE id = $1 AND session_id = $2
            """, key.node_id, key.session_id)

            if row is None:
                return False

            metadata = json.loads(row['metadata'])
            metadata.update(updates)

            await conn.execute("""
                UPDATE context_nodes
                SET metadata = $1::jsonb
                WHERE id = $2 AND session_id = $3
            """, json.dumps(metadata), key.node_id, key.session_id)

            return True

    async def list_keys(
        self,
        session_id: str,
        *,
        tier: Optional[StorageTier] = None,
        node_type: Optional[str] = None,
        limit: int = 1000,
    ) -> list[StorageKey]:
        """List keys for session."""
        await self._ensure_pool()

        query = "SELECT id, version FROM context_nodes WHERE session_id = $1"
        params = [session_id]
        param_idx = 2

        if tier:
            query += f" AND tier = ${param_idx}"
            params.append(tier.value)
            param_idx += 1

        if node_type:
            query += f" AND metadata->>'node_type' = ${param_idx}"
            params.append(node_type)
            param_idx += 1

        query += f" LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [
                StorageKey(
                    session_id=session_id,
                    node_id=row['id'],
                    version=row['version'],
                )
                for row in rows
            ]

    async def search_by_metadata(
        self,
        session_id: str,
        *,
        min_importance: Optional[float] = None,
        tags: Optional[set[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[tuple[StorageKey, StorageMetadata]]:
        """Search by metadata."""
        await self._ensure_pool()

        query = """
            SELECT id, version, metadata, tier, created_at, accessed_at, access_count
            FROM context_nodes
            WHERE session_id = $1
        """
        params = [session_id]
        param_idx = 2

        if min_importance:
            query += f" AND (metadata->>'importance')::float >= ${param_idx}"
            params.append(min_importance)
            param_idx += 1

        if since:
            query += f" AND created_at >= ${param_idx}"
            params.append(since)
            param_idx += 1

        query += f" ORDER BY (metadata->>'importance')::float DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            results = []
            for row in rows:
                key = StorageKey(
                    session_id=session_id,
                    node_id=row['id'],
                    version=row['version'],
                )

                metadata_dict = json.loads(row['metadata'])
                metadata_dict['key'] = key.model_dump(mode="json")
                metadata_dict['tier'] = row['tier']

                metadata = StorageMetadata.model_validate(metadata_dict)

                # Filter by tags if specified
                if tags and not (tags & metadata.tags):
                    continue

                results.append((key, metadata))

            return results

    async def stats(self, session_id: Optional[str] = None) -> StorageStats:
        """Get storage statistics."""
        await self._ensure_pool()

        where_clause = "WHERE session_id = $1" if session_id else ""
        params = [session_id] if session_id else []

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total_items,
                    COALESCE(SUM((metadata->>'size_bytes')::int), 0) as total_size,
                    COALESCE(SUM((metadata->>'token_count')::int), 0) as total_tokens,
                    AVG(access_count) as avg_access,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM context_nodes
                {where_clause}
            """, *params)

            tier_rows = await conn.fetch(f"""
                SELECT tier, COUNT(*) as cnt,
                       SUM((metadata->>'size_bytes')::int) as size
                FROM context_nodes
                {where_clause}
                GROUP BY tier
            """, *params)

            items_by_tier = {r['tier']: r['cnt'] for r in tier_rows}
            size_by_tier = {r['tier']: r['size'] or 0 for r in tier_rows}

            return StorageStats(
                total_items=row['total_items'],
                total_size_bytes=row['total_size'],
                total_tokens=row['total_tokens'],
                items_by_tier=items_by_tier,
                size_by_tier=size_by_tier,
                avg_access_count=row['avg_access'] or 0,
                oldest_item=row['oldest'],
                newest_item=row['newest'],
            )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
```

---

## Component 3: TieredStorage

```python
# context_memory/tiered.py
from typing import Optional
from datetime import datetime, timedelta
from uuid import UUID
import asyncio

from context_core.graph import ContextNode

from .types import StorageKey, StorageMetadata, StorageStats, StorageTier
from .store import MemoryStore


class TierConfig:
    """Configuration for a storage tier."""

    def __init__(
        self,
        tier: StorageTier,
        backend: MemoryStore,
        max_age_seconds: Optional[int] = None,
        max_items: Optional[int] = None,
        min_importance: float = 0.0,
    ):
        self.tier = tier
        self.backend = backend
        self.max_age_seconds = max_age_seconds
        self.max_items = max_items
        self.min_importance = min_importance


class TieredStorage:
    """
    Multi-tier storage with automatic migration.

    Features:
    - Automatic promotion on access
    - Automatic demotion based on age/importance
    - Background migration task
    - Unified query interface
    """

    def __init__(
        self,
        tiers: list[TierConfig],
        promotion_on_access: bool = True,
        migration_interval_seconds: int = 300,
    ):
        self._tiers = {t.tier: t for t in tiers}
        self._tier_order = [StorageTier.HOT, StorageTier.WARM, StorageTier.COLD]
        self._promotion_on_access = promotion_on_access
        self._migration_interval = migration_interval_seconds
        self._migration_task: Optional[asyncio.Task] = None

    async def start_migration_task(self):
        """Start background migration task."""
        self._migration_task = asyncio.create_task(self._migration_loop())

    async def stop_migration_task(self):
        """Stop background migration task."""
        if self._migration_task:
            self._migration_task.cancel()
            try:
                await self._migration_task
            except asyncio.CancelledError:
                pass

    async def _migration_loop(self):
        """Background task for tier migration."""
        while True:
            await asyncio.sleep(self._migration_interval)
            await self._migrate_tiers()

    async def _migrate_tiers(self):
        """Migrate items between tiers based on rules."""
        now = datetime.utcnow()

        # Check each tier for items to demote
        for tier in [StorageTier.HOT, StorageTier.WARM]:
            config = self._tiers.get(tier)
            if not config:
                continue

            next_tier = self._get_next_tier(tier)
            if not next_tier:
                continue

            next_config = self._tiers.get(next_tier)
            if not next_config:
                continue

            # Get items that should be demoted
            # (Implementation would query metadata and move items)
            # This is a simplified version
            pass

    def _get_next_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """Get the next tier for demotion."""
        try:
            idx = self._tier_order.index(tier)
            if idx < len(self._tier_order) - 1:
                return self._tier_order[idx + 1]
        except ValueError:
            pass
        return None

    def _get_prev_tier(self, tier: StorageTier) -> Optional[StorageTier]:
        """Get the previous tier for promotion."""
        try:
            idx = self._tier_order.index(tier)
            if idx > 0:
                return self._tier_order[idx - 1]
        except ValueError:
            pass
        return None

    async def store(
        self,
        node: ContextNode,
        session_id: str,
        tier: StorageTier = StorageTier.HOT,
    ) -> StorageKey:
        """Store node in specified tier."""
        config = self._tiers.get(tier)
        if not config:
            raise ValueError(f"Tier {tier} not configured")

        metadata = StorageMetadata(
            key=StorageKey(session_id=session_id, node_id=node.id, version=1),
            tier=tier,
            size_bytes=0,  # Will be updated by backend
            token_count=node.token_count or 0,
            node_type=node.type.value,
            importance=node.metadata.importance,
            tags=node.metadata.tags,
        )

        return await config.backend.store(node, session_id, metadata)

    async def retrieve(
        self,
        key: StorageKey,
        promote: Optional[bool] = None,
    ) -> Optional[ContextNode]:
        """
        Retrieve node, checking all tiers.

        If found in a lower tier and promote=True (or default),
        promotes to HOT tier.
        """
        promote = promote if promote is not None else self._promotion_on_access

        # Check tiers in order
        for tier in self._tier_order:
            config = self._tiers.get(tier)
            if not config:
                continue

            node = await config.backend.retrieve(key)
            if node:
                # Promote if needed
                if promote and tier != StorageTier.HOT:
                    hot_config = self._tiers.get(StorageTier.HOT)
                    if hot_config:
                        # Store in HOT
                        await hot_config.backend.store(node, key.session_id)
                        # Optionally remove from lower tier
                        await config.backend.delete(key)

                return node

        return None

    async def delete(self, key: StorageKey) -> bool:
        """Delete from all tiers."""
        deleted = False
        for tier in self._tier_order:
            config = self._tiers.get(tier)
            if config:
                if await config.backend.delete(key):
                    deleted = True
        return deleted

    async def get_metadata(
        self,
        key: StorageKey,
    ) -> Optional[StorageMetadata]:
        """Get metadata from first tier containing key."""
        for tier in self._tier_order:
            config = self._tiers.get(tier)
            if not config:
                continue

            metadata = await config.backend.get_metadata(key)
            if metadata:
                return metadata

        return None

    async def migrate_to_tier(
        self,
        key: StorageKey,
        target_tier: StorageTier,
    ) -> bool:
        """Manually migrate an item to a specific tier."""
        # Find current tier
        current_tier = None
        node = None

        for tier in self._tier_order:
            config = self._tiers.get(tier)
            if not config:
                continue

            node = await config.backend.retrieve(key)
            if node:
                current_tier = tier
                break

        if not node or not current_tier:
            return False

        if current_tier == target_tier:
            return True

        # Store in target tier
        target_config = self._tiers.get(target_tier)
        if not target_config:
            return False

        await target_config.backend.store(node, key.session_id)

        # Remove from current tier
        current_config = self._tiers.get(current_tier)
        if current_config:
            await current_config.backend.delete(key)

        return True

    async def stats(self) -> dict[str, StorageStats]:
        """Get stats for all tiers."""
        result = {}
        for tier, config in self._tiers.items():
            result[tier.value] = await config.backend.stats()
        return result

    async def close(self):
        """Close all backends."""
        await self.stop_migration_task()
        for config in self._tiers.values():
            await config.backend.close()
```

---

## Component 4: MemoryRetriever

### 4.1 RetrievalStrategy Protocol

```python
# context_memory/retrieval/base.py
from typing import Protocol, Optional, runtime_checkable
from uuid import UUID

from context_core.graph import ContextGraph, ContextNode

from ..types import RetrievalResult, StorageTier
from ..store import MemoryStore


class RetrievalQuery(BaseModel):
    """Query parameters for retrieval."""
    session_id: str
    query_text: Optional[str] = None
    entity_ids: Optional[list[str]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    node_types: Optional[list[str]] = None
    min_importance: Optional[float] = None
    max_results: int = 10


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Protocol for retrieval strategies."""

    @property
    def name(self) -> str:
        """Strategy identifier."""
        ...

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant context from memory.

        Args:
            query: Retrieval query parameters
            store: Memory store to query
            current_context: Current conversation context for relevance

        Returns:
            List of retrieval results with scores
        """
        ...

    def score(self, result: RetrievalResult, query: RetrievalQuery) -> float:
        """Score a result for ranking."""
        ...
```

### 4.2 SemanticRetrieval

```python
# context_memory/retrieval/semantic.py
from typing import Optional
import time

from context_core.graph import ContextGraph, ContextNode
from context_core.semantic import SemanticIndex

from ..types import RetrievalResult, StorageTier
from ..store import MemoryStore
from .base import RetrievalStrategy, RetrievalQuery


class SemanticRetrieval:
    """
    Retrieve context using semantic similarity.

    Uses embeddings to find contextually relevant past interactions.
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        min_similarity: float = 0.6,
    ):
        self._semantic_index = semantic_index
        self._min_similarity = min_similarity

    @property
    def name(self) -> str:
        return "semantic"

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using semantic similarity."""
        if not query.query_text:
            return []

        start = time.perf_counter()

        # Search semantic index
        search_results = self._semantic_index.search(
            query=query.query_text,
            k=query.max_results * 2,  # Get more for filtering
            min_score=self._min_similarity,
        )

        results = []
        for sr in search_results:
            # Get full node from store
            from ..types import StorageKey
            key = StorageKey(
                session_id=query.session_id,
                node_id=sr.id,
                version=1,
            )

            node = await store.retrieve(key)
            if not node:
                continue

            # Apply filters
            if query.node_types and node.type.value not in query.node_types:
                continue

            if query.min_importance and node.metadata.importance < query.min_importance:
                continue

            latency = (time.perf_counter() - start) * 1000

            results.append(RetrievalResult(
                node=node,
                score=sr.score,
                source_tier=StorageTier.WARM,
                retrieval_method=self.name,
                latency_ms=latency,
            ))

            if len(results) >= query.max_results:
                break

        return results

    def score(self, result: RetrievalResult, query: RetrievalQuery) -> float:
        """Score is the similarity score."""
        return result.score
```

### 4.3 EntityRetrieval

```python
# context_memory/retrieval/entity.py
from typing import Optional
import time

from context_core.graph import ContextGraph, ContextNode
from context_core.entities import EntityTracker

from ..types import RetrievalResult, StorageTier
from ..store import MemoryStore
from .base import RetrievalStrategy, RetrievalQuery


class EntityRetrieval:
    """
    Retrieve context based on entity mentions.

    Finds past context mentioning the same entities.
    """

    def __init__(
        self,
        entity_tracker: EntityTracker,
    ):
        self._entity_tracker = entity_tracker

    @property
    def name(self) -> str:
        return "entity"

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """Retrieve based on entity mentions."""
        if not query.entity_ids:
            # Extract entities from query text if provided
            if query.query_text:
                from uuid import uuid4
                temp_node_id = uuid4()
                entities = self._entity_tracker.extract_from_text(
                    query.query_text,
                    temp_node_id,
                )
                query.entity_ids = [str(e.id) for e in entities]

        if not query.entity_ids:
            return []

        start = time.perf_counter()
        results = []

        # Find nodes mentioning these entities
        for entity_id in query.entity_ids:
            entity = self._entity_tracker.get_entity(UUID(entity_id))
            if not entity:
                continue

            for node_id in entity.node_ids:
                from ..types import StorageKey
                key = StorageKey(
                    session_id=query.session_id,
                    node_id=node_id,
                    version=1,
                )

                node = await store.retrieve(key)
                if not node:
                    continue

                # Calculate score based on entity importance and recency
                entity_importance = entity.importance
                mention_count = entity.mention_count
                score = 0.5 * entity_importance + 0.5 * min(mention_count / 10, 1.0)

                latency = (time.perf_counter() - start) * 1000

                results.append(RetrievalResult(
                    node=node,
                    score=score,
                    source_tier=StorageTier.WARM,
                    retrieval_method=self.name,
                    latency_ms=latency,
                ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:query.max_results]

    def score(self, result: RetrievalResult, query: RetrievalQuery) -> float:
        return result.score
```

### 4.4 EnsembleRetriever

```python
# context_memory/retrieval/ensemble.py
from typing import Optional
from collections import defaultdict

from context_core.graph import ContextGraph, ContextNode

from ..types import RetrievalResult
from ..store import MemoryStore
from .base import RetrievalStrategy, RetrievalQuery


class EnsembleRetriever:
    """
    Combine multiple retrieval strategies with weighted scoring.

    Uses reciprocal rank fusion for combining rankings.
    """

    def __init__(
        self,
        strategies: list[tuple[RetrievalStrategy, float]],  # (strategy, weight)
        k: int = 60,  # RRF constant
    ):
        self._strategies = strategies
        self._k = k

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]:
        """Retrieve using all strategies and combine."""
        # Collect results from all strategies
        all_results: dict[UUID, list[tuple[RetrievalResult, float, int]]] = defaultdict(list)

        for strategy, weight in self._strategies:
            results = await strategy.retrieve(query, store, current_context)

            for rank, result in enumerate(results):
                node_id = result.node.id
                all_results[node_id].append((result, weight, rank))

        # Combine using reciprocal rank fusion
        combined_scores: dict[UUID, float] = {}
        best_results: dict[UUID, RetrievalResult] = {}

        for node_id, result_list in all_results.items():
            score = 0.0
            for result, weight, rank in result_list:
                # RRF score
                score += weight * (1.0 / (self._k + rank + 1))

                if node_id not in best_results:
                    best_results[node_id] = result

            combined_scores[node_id] = score

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build final results
        final_results = []
        for node_id in sorted_ids[:query.max_results]:
            result = best_results[node_id]
            # Update score with combined score
            result.score = combined_scores[node_id]
            result.retrieval_method = "ensemble"
            final_results.append(result)

        return final_results
```

---

## Component 5: ArtifactManager

```python
# context_memory/artifacts/manager.py
from typing import Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
import hashlib
import json
from pydantic import BaseModel, Field

from ..store import MemoryStore
from ..types import StorageKey


class ArtifactVersion(BaseModel):
    """A specific version of an artifact."""
    version: int
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Artifact(BaseModel):
    """A versioned artifact."""
    id: UUID = Field(default_factory=uuid4)
    session_id: str
    artifact_type: str          # "code", "file", "data", etc.
    name: str                   # Human-readable identifier
    current_version: int = 1
    versions: list[ArtifactVersion] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ArtifactManager:
    """
    Manage versioned artifacts with content addressing.

    Features:
    - Content-addressed storage
    - Version history
    - Diff between versions
    - Type-specific handling
    """

    def __init__(
        self,
        store: MemoryStore,
    ):
        self._store = store
        self._artifacts: dict[UUID, Artifact] = {}
        self._content_cache: dict[str, bytes] = {}

    def _content_hash(self, content: bytes) -> str:
        """Generate content hash."""
        return hashlib.sha256(content).hexdigest()

    async def create_artifact(
        self,
        session_id: str,
        name: str,
        content: bytes | str,
        artifact_type: str = "data",
        metadata: Optional[dict] = None,
    ) -> Artifact:
        """Create a new artifact."""
        if isinstance(content, str):
            content = content.encode()

        content_hash = self._content_hash(content)

        version = ArtifactVersion(
            version=1,
            content_hash=content_hash,
            size_bytes=len(content),
            metadata=metadata or {},
        )

        artifact = Artifact(
            session_id=session_id,
            artifact_type=artifact_type,
            name=name,
            versions=[version],
        )

        # Store content
        self._content_cache[content_hash] = content
        self._artifacts[artifact.id] = artifact

        return artifact

    async def update_artifact(
        self,
        artifact_id: UUID,
        content: bytes | str,
        metadata: Optional[dict] = None,
    ) -> Optional[ArtifactVersion]:
        """Add a new version to an artifact."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return None

        if isinstance(content, str):
            content = content.encode()

        content_hash = self._content_hash(content)

        # Check if content actually changed
        if artifact.versions and artifact.versions[-1].content_hash == content_hash:
            return artifact.versions[-1]

        new_version = ArtifactVersion(
            version=artifact.current_version + 1,
            content_hash=content_hash,
            size_bytes=len(content),
            metadata=metadata or {},
        )

        artifact.versions.append(new_version)
        artifact.current_version = new_version.version
        artifact.updated_at = datetime.utcnow()

        self._content_cache[content_hash] = content

        return new_version

    async def get_artifact(
        self,
        artifact_id: UUID,
        version: Optional[int] = None,
    ) -> Optional[tuple[Artifact, bytes]]:
        """Get artifact and its content."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return None

        if version:
            target_version = next(
                (v for v in artifact.versions if v.version == version),
                None
            )
        else:
            target_version = artifact.versions[-1] if artifact.versions else None

        if not target_version:
            return None

        content = self._content_cache.get(target_version.content_hash)
        if not content:
            return None

        return artifact, content

    async def diff_versions(
        self,
        artifact_id: UUID,
        version1: int,
        version2: int,
    ) -> Optional[dict]:
        """Generate diff between two versions."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return None

        v1 = next((v for v in artifact.versions if v.version == version1), None)
        v2 = next((v for v in artifact.versions if v.version == version2), None)

        if not v1 or not v2:
            return None

        content1 = self._content_cache.get(v1.content_hash, b"")
        content2 = self._content_cache.get(v2.content_hash, b"")

        try:
            text1 = content1.decode()
            text2 = content2.decode()

            import difflib
            diff = list(difflib.unified_diff(
                text1.splitlines(keepends=True),
                text2.splitlines(keepends=True),
                fromfile=f"v{version1}",
                tofile=f"v{version2}",
            ))

            return {
                "type": "unified_diff",
                "lines": diff,
                "additions": sum(1 for l in diff if l.startswith("+")),
                "deletions": sum(1 for l in diff if l.startswith("-")),
            }
        except UnicodeDecodeError:
            # Binary content
            return {
                "type": "binary",
                "size_change": len(content2) - len(content1),
            }

    async def list_artifacts(
        self,
        session_id: str,
        artifact_type: Optional[str] = None,
    ) -> list[Artifact]:
        """List artifacts for a session."""
        results = []
        for artifact in self._artifacts.values():
            if artifact.session_id != session_id:
                continue
            if artifact_type and artifact.artifact_type != artifact_type:
                continue
            results.append(artifact)
        return results

    async def delete_artifact(
        self,
        artifact_id: UUID,
    ) -> bool:
        """Delete an artifact and all versions."""
        artifact = self._artifacts.pop(artifact_id, None)
        if not artifact:
            return False

        # Clean up content (only if not referenced elsewhere)
        for version in artifact.versions:
            referenced = any(
                v.content_hash == version.content_hash
                for a in self._artifacts.values()
                for v in a.versions
            )
            if not referenced:
                self._content_cache.pop(version.content_hash, None)

        return True
```

---

## Component 6: WorkingMemory

```python
# context_memory/working.py
from typing import Optional
from uuid import UUID
from collections import OrderedDict
from datetime import datetime
import asyncio

from context_core.graph import ContextGraph, ContextNode

from .types import StorageKey, StorageMetadata, StorageTier
from .store import MemoryStore


class WorkingMemory:
    """
    Fast-access cache for active context.

    Features:
    - LRU eviction
    - Sync with backing store
    - Token-aware capacity management
    - Automatic persistence
    """

    def __init__(
        self,
        backing_store: MemoryStore,
        max_tokens: int = 50000,
        max_items: int = 1000,
        sync_interval_seconds: int = 60,
    ):
        self._backing_store = backing_store
        self._max_tokens = max_tokens
        self._max_items = max_items
        self._sync_interval = sync_interval_seconds

        self._cache: OrderedDict[UUID, ContextNode] = OrderedDict()
        self._metadata: dict[UUID, StorageMetadata] = {}
        self._dirty: set[UUID] = set()  # Modified but not synced
        self._current_tokens = 0

        self._sync_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start_sync_task(self):
        """Start background sync task."""
        self._sync_task = asyncio.create_task(self._sync_loop())

    async def stop_sync_task(self):
        """Stop background sync and flush."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def _sync_loop(self):
        """Background sync to backing store."""
        while True:
            await asyncio.sleep(self._sync_interval)
            await self.flush()

    async def add(
        self,
        node: ContextNode,
        session_id: str,
    ) -> None:
        """Add a node to working memory."""
        async with self._lock:
            node_tokens = node.token_count or 0

            # Evict if needed
            while (
                self._current_tokens + node_tokens > self._max_tokens or
                len(self._cache) >= self._max_items
            ):
                if not self._cache:
                    break
                await self._evict_one()

            # Add to cache
            self._cache[node.id] = node
            self._cache.move_to_end(node.id)
            self._current_tokens += node_tokens
            self._dirty.add(node.id)

            # Create metadata
            self._metadata[node.id] = StorageMetadata(
                key=StorageKey(session_id=session_id, node_id=node.id, version=1),
                tier=StorageTier.HOT,
                size_bytes=0,
                token_count=node_tokens,
                node_type=node.type.value,
                importance=node.metadata.importance,
                tags=node.metadata.tags,
            )

    async def get(self, node_id: UUID) -> Optional[ContextNode]:
        """Get a node from working memory."""
        async with self._lock:
            node = self._cache.get(node_id)
            if node:
                # Move to end (most recently used)
                self._cache.move_to_end(node_id)
                self._metadata[node_id].touch()
            return node

    async def remove(self, node_id: UUID) -> Optional[ContextNode]:
        """Remove a node from working memory."""
        async with self._lock:
            node = self._cache.pop(node_id, None)
            if node:
                self._current_tokens -= node.token_count or 0
                self._metadata.pop(node_id, None)
                self._dirty.discard(node_id)
            return node

    async def _evict_one(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return

        # Get LRU item (first in OrderedDict)
        node_id, node = next(iter(self._cache.items()))

        # Persist if dirty
        if node_id in self._dirty:
            metadata = self._metadata.get(node_id)
            if metadata:
                await self._backing_store.store(
                    node,
                    metadata.key.session_id,
                    metadata,
                )
            self._dirty.discard(node_id)

        # Remove from cache
        del self._cache[node_id]
        self._metadata.pop(node_id, None)
        self._current_tokens -= node.token_count or 0

    async def flush(self) -> int:
        """Persist all dirty items to backing store."""
        async with self._lock:
            flushed = 0
            dirty_ids = list(self._dirty)

            for node_id in dirty_ids:
                node = self._cache.get(node_id)
                metadata = self._metadata.get(node_id)

                if node and metadata:
                    await self._backing_store.store(
                        node,
                        metadata.key.session_id,
                        metadata,
                    )
                    flushed += 1

            self._dirty.clear()
            return flushed

    async def load_from_store(
        self,
        session_id: str,
        limit: int = 100,
    ) -> int:
        """Load recent items from backing store."""
        keys = await self._backing_store.list_keys(session_id, limit=limit)
        loaded = 0

        for key in keys:
            if len(self._cache) >= self._max_items:
                break

            node = await self._backing_store.retrieve(key)
            if node and node.id not in self._cache:
                async with self._lock:
                    self._cache[node.id] = node
                    self._current_tokens += node.token_count or 0

                    metadata = await self._backing_store.get_metadata(key)
                    if metadata:
                        self._metadata[node.id] = metadata

                    loaded += 1

        return loaded

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "items": len(self._cache),
            "tokens": self._current_tokens,
            "max_tokens": self._max_tokens,
            "dirty_items": len(self._dirty),
            "utilization": self._current_tokens / self._max_tokens if self._max_tokens else 0,
        }
```

---

## Task Breakdown

### Week 13-14: Storage Layer

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| MS-001 | Create `types.py` with all types | 3 | None | All models validate |
| MS-002 | Define `MemoryStore` protocol | 3 | MS-001 | Protocol complete |
| MS-003 | Implement `FileSystemStore` | 8 | MS-002 | All CRUD operations work |
| MS-004 | Implement `PostgresStore` | 10 | MS-002 | PostgreSQL backend works |
| MS-005 | Implement `RedisStore` | 6 | MS-002 | Redis backend works |
| MS-006 | Write unit tests for backends | 8 | MS-003 to MS-005 | 90%+ coverage |

### Week 15: Tiered Storage

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| TS-001 | Implement `TierConfig` | 2 | MS-001 | Config model works |
| TS-002 | Implement `TieredStorage` core | 8 | TS-001 | Multi-tier storage works |
| TS-003 | Implement tier migration | 6 | TS-002 | Auto-migration works |
| TS-004 | Implement promotion on access | 3 | TS-002 | Promotion works |
| TS-005 | Write unit tests | 6 | TS-001 to TS-004 | 90%+ coverage |

### Week 16: Memory Retrieval

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| MR-001 | Define `RetrievalStrategy` protocol | 2 | None | Protocol complete |
| MR-002 | Implement `SemanticRetrieval` | 6 | MR-001 | Semantic search works |
| MR-003 | Implement `EntityRetrieval` | 5 | MR-001 | Entity-based retrieval works |
| MR-004 | Implement `TemporalRetrieval` | 4 | MR-001 | Time-based retrieval works |
| MR-005 | Implement `EnsembleRetriever` | 6 | MR-002 to MR-004 | Ensemble ranking works |
| MR-006 | Write unit tests | 6 | MR-001 to MR-005 | 90%+ coverage |

### Week 17: Artifacts & Working Memory

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| AM-001 | Implement `ArtifactVersion` | 2 | None | Model works |
| AM-002 | Implement `ArtifactManager` | 8 | AM-001 | Versioning works |
| AM-003 | Implement diff functionality | 4 | AM-002 | Diffs generated |
| WM-001 | Implement `WorkingMemory` | 8 | MS-002 | LRU cache works |
| WM-002 | Implement background sync | 4 | WM-001 | Auto-sync works |
| WM-003 | Write unit tests | 6 | AM-001 to WM-002 | 90%+ coverage |

### Week 18: Integration

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| INT-001 | Create public API | 3 | All above | Clean exports |
| INT-002 | Write integration tests | 8 | INT-001 | Full stack works |
| INT-003 | Performance benchmarks | 4 | INT-001 | p99 < 100ms retrieval |
| INT-004 | Documentation | 6 | INT-001 | API docs complete |

---

## Test Specifications

### Performance Requirements

| Operation | p50 | p99 | Target |
|-----------|-----|-----|--------|
| Store (hot tier) | < 5ms | < 20ms | Pass |
| Retrieve (hot tier) | < 2ms | < 10ms | Pass |
| Retrieve (warm tier) | < 20ms | < 100ms | Pass |
| Semantic search | < 50ms | < 200ms | Pass |
| Ensemble retrieval | < 100ms | < 500ms | Pass |

### Example Tests

```python
# tests/test_tiered_storage.py
import pytest
import asyncio
from context_core.graph import ContextNode, NodeType, Content
from context_memory import TieredStorage, TierConfig, StorageTier
from context_memory.backends import FileSystemStore

class TestTieredStorage:
    @pytest.fixture
    async def tiered_storage(self, tmp_path):
        hot = FileSystemStore(tmp_path / "hot")
        warm = FileSystemStore(tmp_path / "warm")
        cold = FileSystemStore(tmp_path / "cold")

        storage = TieredStorage([
            TierConfig(StorageTier.HOT, hot, max_age_seconds=60),
            TierConfig(StorageTier.WARM, warm, max_age_seconds=3600),
            TierConfig(StorageTier.COLD, cold),
        ])

        yield storage
        await storage.close()

    async def test_store_in_hot_tier(self, tiered_storage):
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="test"),
        )

        key = await tiered_storage.store(node, "session1")
        metadata = await tiered_storage.get_metadata(key)

        assert metadata.tier == StorageTier.HOT

    async def test_promotion_on_access(self, tiered_storage):
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="test"),
        )

        # Store in warm tier
        key = await tiered_storage.store(node, "session1", tier=StorageTier.WARM)

        # Access should promote to hot
        retrieved = await tiered_storage.retrieve(key, promote=True)
        metadata = await tiered_storage.get_metadata(key)

        assert retrieved is not None
        assert metadata.tier == StorageTier.HOT


class TestWorkingMemory:
    async def test_lru_eviction(self, tmp_path):
        store = FileSystemStore(tmp_path)
        wm = WorkingMemory(store, max_tokens=100, max_items=5)

        # Add 6 items to trigger eviction
        for i in range(6):
            node = ContextNode(
                type=NodeType.MESSAGE,
                content=Content(text=f"msg{i}"),
                token_count=20,
            )
            await wm.add(node, "session1")

        assert wm.stats["items"] == 5
        # First item should be evicted and persisted
```

---

## Definition of Done for Phase 3

1. **All storage backends implemented** and tested
2. **Tiered storage** with automatic migration
3. **4 retrieval strategies** working with ensemble
4. **Artifact manager** with versioning
5. **Working memory** with LRU eviction
6. **Unit test coverage** >= 90%
7. **Performance targets** met:
   - Retrieval p99 < 100ms
   - 10,000 items stored in < 30s
8. **Documentation** complete

---

*This specification provides complete storage layer implementation. Each component can be developed and tested independently.*
