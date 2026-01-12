"""Lossless compression strategies.

Lossless strategies compress content without losing any information.
The original content can always be fully recovered.

Strategies:
- ExternalizePayloads: Store large payloads externally
- DeduplicateSemantically: Remove semantic duplicates
- CollapseToolChains: Collapse sequential tool calls
"""

from __future__ import annotations

from context_compression.strategies.lossless.collapse import CollapseToolChains
from context_compression.strategies.lossless.deduplicate import DeduplicateSemantically
from context_compression.strategies.lossless.externalize import (
    ExternalizePayloads,
    ExternalStorage,
    FileSystemExternalStorage,
    InMemoryExternalStorage,
)

__all__ = [
    "ExternalStorage",
    "FileSystemExternalStorage",
    "InMemoryExternalStorage",
    "ExternalizePayloads",
    "DeduplicateSemantically",
    "CollapseToolChains",
]
