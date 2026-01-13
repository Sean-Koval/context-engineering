"""Storage backends for context-memory.

This module provides concrete implementations of the MemoryStore protocol
for various storage systems.

Available Backends:
    FileSystemStore: Local filesystem-based storage for development and
        single-node deployments.
    SQLiteStore: Embedded SQLite database for lightweight deployments
        with better query performance than filesystem.
    PostgresStore: Production-grade PostgreSQL backend with connection pooling
        for concurrent access and scalability.
    RedisStore: High-performance Redis backend for hot-tier caching with
        TTL-based expiration and sub-10ms latency.
"""

from __future__ import annotations

from context_memory.backends.filesystem import FileSystemStore
from context_memory.backends.postgres import PostgresStore
from context_memory.backends.redis import RedisStore
from context_memory.backends.sqlite import SQLiteStore

__all__ = [
    "FileSystemStore",
    "PostgresStore",
    "RedisStore",
    "SQLiteStore",
]
