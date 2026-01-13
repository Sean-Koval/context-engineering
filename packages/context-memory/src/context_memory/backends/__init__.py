"""Storage backends for context-memory.

This module provides concrete implementations of the MemoryStore protocol
for various storage systems.

Available Backends:
    FileSystemStore: Local filesystem-based storage for development and
        single-node deployments.
    SQLiteStore: Embedded SQLite database for lightweight deployments
        with better query performance than filesystem.
"""

from __future__ import annotations

from context_memory.backends.filesystem import FileSystemStore
from context_memory.backends.sqlite import SQLiteStore

__all__ = [
    "FileSystemStore",
    "SQLiteStore",
]
