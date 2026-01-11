"""Compression strategies module.

This module provides the CompressionStrategy protocol and base implementations
for the three strategy tiers: lossless, compaction, and summarization.
"""

from __future__ import annotations

from context_compression.strategies.base import (
    BaseCompressionStrategy,
    CompressionStrategy,
)

__all__ = [
    "CompressionStrategy",
    "BaseCompressionStrategy",
]
