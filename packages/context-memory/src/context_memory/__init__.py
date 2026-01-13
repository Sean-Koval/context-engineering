"""Context Memory - Persistent storage and retrieval for ContextEngine."""

from __future__ import annotations

from context_memory.backends import (
    FileSystemStore,
    PostgresStore,
    RedisStore,
    SQLiteStore,
)
from context_memory.store import MemoryStore
from context_memory.types import (
    RetrievalResult,
    StorageKey,
    StorageMetadata,
    StorageStats,
    StorageTier,
)

__all__ = [
    # Protocol
    "MemoryStore",
    # Backends
    "FileSystemStore",
    "PostgresStore",
    "RedisStore",
    "SQLiteStore",
    # Types
    "StorageTier",
    "StorageKey",
    "StorageMetadata",
    "StorageStats",
    "RetrievalResult",
]
