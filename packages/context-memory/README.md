# context-memory

> Persistent storage, tiered caching, and intelligent retrieval for ContextEngine

## Installation

```bash
uv pip install -e packages/context-memory
```

## Components

### TieredStorage

Hot/warm/cold storage with automatic migration between tiers.

```python
from context_memory import TieredStorage, TierConfig, StorageTier

# Configure tiers
config = {
    StorageTier.HOT: TierConfig(
        max_items=1000,
        max_bytes=100 * 1024 * 1024,  # 100MB
        ttl_seconds=3600,  # 1 hour
    ),
    StorageTier.WARM: TierConfig(
        max_items=10000,
        max_bytes=1024 * 1024 * 1024,  # 1GB
        ttl_seconds=86400,  # 24 hours
    ),
    StorageTier.COLD: TierConfig(
        max_items=None,  # Unlimited
        max_bytes=None,
    ),
}

# Create tiered storage
storage = TieredStorage(config)

# Store items (automatically placed in hot tier)
await storage.store(key, metadata, content)

# Retrieve (promotes to hotter tier on access)
item = await storage.retrieve(key)

# Items automatically migrate based on access patterns
```

### WorkingMemory

LRU cache with background sync to persistent storage.

```python
from context_memory import WorkingMemory

# Create working memory with limits
memory = WorkingMemory(
    store=persistent_store,
    max_tokens=50000,
    max_items=500,
    sync_interval=30.0,  # Sync every 30 seconds
)

async with memory:
    # Add nodes (automatically syncs when dirty)
    await memory.add(node)

    # Get with LRU tracking
    node = await memory.get(node_id)

    # Evicts least-recently-used when limits exceeded
    stats = memory.stats
    print(f"Token utilization: {stats.token_utilization:.1%}")
```

### Storage Backends

Multiple backend implementations for different use cases.

```python
from context_memory import FileSystemStore, SQLiteStore, PostgresStore, RedisStore

# File system (development)
fs_store = FileSystemStore(base_path="./data")

# SQLite (single-node production)
sqlite_store = SQLiteStore(db_path="./context.db")

# PostgreSQL (multi-node production)
pg_store = PostgresStore(connection_string="postgresql://...")

# Redis (high-performance caching)
redis_store = RedisStore(url="redis://localhost:6379")
```

### Retrieval Strategies

Multiple retrieval methods with ensemble combining.

```python
from context_memory import (
    EnsembleRetriever,
    SemanticRetrieval,
    EntityRetrieval,
    TemporalRetrieval,
    RetrievalQuery,
)

# Create retrievers
semantic = SemanticRetrieval(semantic_index)
entity = EntityRetrieval(entity_tracker)
temporal = TemporalRetrieval()

# Combine with ensemble
retriever = EnsembleRetriever(
    strategies=[
        (semantic, 0.5),   # 50% weight
        (entity, 0.3),     # 30% weight
        (temporal, 0.2),   # 20% weight
    ]
)

# Query
query = RetrievalQuery(
    text="authentication implementation",
    top_k=10,
    min_score=0.5,
)
results = await retriever.retrieve(query)
```

### Artifact Management

Versioned artifact storage with diffing support.

```python
from context_memory import ArtifactManager, Artifact

manager = ArtifactManager(store=storage)

# Store artifact with automatic versioning
artifact = Artifact(
    name="auth.py",
    content="def authenticate(user): ...",
    artifact_type="code",
)
version = await manager.store(artifact)

# Get specific version
artifact = await manager.get("auth.py", version=2)

# Get diff between versions
diff = await manager.diff("auth.py", version_a=1, version_b=2)
print(diff.additions, diff.deletions)

# List all versions
versions = await manager.list_versions("auth.py")
```

### Eviction Management

Intelligent eviction with importance scoring.

```python
from context_memory import (
    EvictionManager,
    MultiTierEvictionManager,
    LRUImportanceScorer,
    CapacityConfig,
)

# Create scorer (combines recency and importance)
scorer = LRUImportanceScorer(
    recency_weight=0.6,
    importance_weight=0.4,
)

# Create eviction manager
eviction = EvictionManager(
    store=storage,
    scorer=scorer,
    config=CapacityConfig(
        max_items=1000,
        max_bytes=100 * 1024 * 1024,
        target_utilization=0.8,  # Evict when >80% full
    ),
)

# Run eviction (returns evicted items)
result = await eviction.evict()
print(f"Evicted {result.items_evicted} items, freed {result.bytes_freed} bytes")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WorkingMemory                            │
│                   (LRU Cache + Sync)                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     TieredStorage                            │
├─────────────────┬─────────────────┬─────────────────────────┤
│       HOT       │      WARM       │         COLD            │
│   (In-memory)   │    (SQLite)     │    (FileSystem)         │
│   < 1 hour      │   < 24 hours    │     Permanent           │
└─────────────────┴─────────────────┴─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   EnsembleRetriever                          │
├─────────────────┬─────────────────┬─────────────────────────┤
│    Semantic     │     Entity      │       Temporal          │
│  (Embeddings)   │    (NER)        │    (Time-based)         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Tests

```bash
# Run all tests
uv run pytest packages/context-memory/tests -v

# Run with coverage
uv run pytest packages/context-memory/tests --cov=context_memory
```

**307 tests** covering:
- Storage backends (FileSystem, SQLite, Postgres, Redis)
- Tiered storage with migration
- Working memory with eviction
- All retrieval strategies
- Artifact versioning and diffing
- Eviction with importance scoring
