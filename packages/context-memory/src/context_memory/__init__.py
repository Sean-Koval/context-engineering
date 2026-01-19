"""Context Memory - Persistent storage and retrieval for ContextEngine."""

from __future__ import annotations

from context_memory.artifacts import (
    Artifact,
    ArtifactManager,
    ArtifactVersion,
    DiffResult,
)
from context_memory.backends import (
    FileSystemStore,
    PostgresStore,
    RedisStore,
    SQLiteStore,
)
from context_memory.eviction import (
    CapacityConfig,
    EvictionCandidate,
    EvictionManager,
    EvictionResult,
    LRUImportanceScorer,
    MultiTierEvictionManager,
    PureAccessCountScorer,
    TierEvictionConfig,
)
from context_memory.migration import (
    MigrationManager,
    MigrationPolicy,
    MigrationStats,
    TierMigrationConfig,
    TierMigrationCoordinator,
)
from context_memory.retrieval import (
    EnsembleRetriever,
    EntityRetrieval,
    MemoryRetriever,
    RetrievalQuery,
    RetrievalStrategy,
    SemanticRetrieval,
    TemporalRetrieval,
)
from context_memory.store import MemoryStore
from context_memory.tiered import MigrationResult, TierConfig, TieredStorage
from context_memory.types import (
    RetrievalResult,
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)
from context_memory.working import WorkingMemory, WorkingMemoryStats

__all__ = [
    # Protocol
    "MemoryStore",
    # Backends
    "FileSystemStore",
    "PostgresStore",
    "RedisStore",
    "SQLiteStore",
    # Tiered Storage
    "TierConfig",
    "TieredStorage",
    "MigrationResult",
    # Eviction
    "EvictionManager",
    "MultiTierEvictionManager",
    "EvictionResult",
    "EvictionCandidate",
    "CapacityConfig",
    "TierEvictionConfig",
    "LRUImportanceScorer",
    "PureAccessCountScorer",
    # Migration
    "MigrationManager",
    "MigrationPolicy",
    "MigrationStats",
    "TierMigrationConfig",
    "TierMigrationCoordinator",
    # Retrieval
    "RetrievalQuery",
    "RetrievalStrategy",
    "SemanticRetrieval",
    "EntityRetrieval",
    "TemporalRetrieval",
    "EnsembleRetriever",
    "MemoryRetriever",
    # Artifacts
    "Artifact",
    "ArtifactManager",
    "ArtifactVersion",
    "DiffResult",
    # Working Memory
    "WorkingMemory",
    "WorkingMemoryStats",
    # Types
    "StorageTier",
    "StorageKey",
    "StorageMetadata",
    "StorageStats",
    "RetrievalResult",
]
