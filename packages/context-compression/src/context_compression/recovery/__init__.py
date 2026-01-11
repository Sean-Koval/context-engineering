"""Recovery module for tracking and reversing compression operations.

This module provides:
- Operation types for each kind of compression
- RecoveryManifest for tracking all operations in a session
"""

from __future__ import annotations

from context_compression.recovery.manifest import RecoveryManifest
from context_compression.recovery.operations import (
    CollapseOperation,
    CompactOperation,
    CompressionOperation,
    DeduplicateOperation,
    EvictOperation,
    ExternalizeOperation,
    SummarizeOperation,
)

__all__ = [
    "RecoveryManifest",
    "CompressionOperation",
    "ExternalizeOperation",
    "DeduplicateOperation",
    "CollapseOperation",
    "CompactOperation",
    "SummarizeOperation",
    "EvictOperation",
]
