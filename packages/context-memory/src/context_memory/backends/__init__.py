"""Storage backends for context-memory.

This module provides concrete implementations of the MemoryStore protocol
for various storage systems.

Available Backends:
    FileSystemStore: Local filesystem-based storage for development and
        single-node deployments.
"""

from __future__ import annotations

from context_memory.backends.filesystem import FileSystemStore

__all__ = [
    "FileSystemStore",
]
