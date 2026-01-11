"""Context Compression - Compression pipeline for ContextEngine.

This package provides a multi-strategy compression pipeline for reducing
context token usage while preserving important information.

Key components:
- CompressionPipeline: Orchestrates compression strategies
- CompressionStrategy: Protocol for implementing strategies
- RecoveryManifest: Tracks operations for potential recovery

Compression tiers (in execution order):
1. LOSSLESS: Fully reversible (externalize, deduplicate, collapse)
2. COMPACTION: Mostly reversible (schema, entity-centric, task-relevance)
3. SUMMARIZATION: Irreversible (hierarchical, task-aware)

Example:
    >>> from context_compression import CompressionPipeline
    >>> from context_compression.strategies.lossless import (
    ...     ExternalizePayloads,
    ...     InMemoryExternalStorage,
    ... )
    >>>
    >>> storage = InMemoryExternalStorage()
    >>> pipeline = CompressionPipeline()
    >>> pipeline.register_strategy(ExternalizePayloads(storage))
    >>>
    >>> results = pipeline.compress(graph, target_tokens=5000)
    >>> print(f"Saved {sum(r.tokens_saved for r in results)} tokens")
"""

from __future__ import annotations

from context_compression.pipeline import CompressionPipeline
from context_compression.recovery import RecoveryManifest
from context_compression.strategies import (
    BaseCompressionStrategy,
    CompressionStrategy,
)
from context_compression.types import (
    CompressionPlan,
    CompressionResult,
    CompressionTier,
    PipelineConfig,
    PreservationRule,
)

__all__ = [
    # Pipeline
    "CompressionPipeline",
    # Strategy
    "CompressionStrategy",
    "BaseCompressionStrategy",
    # Types
    "CompressionTier",
    "CompressionResult",
    "CompressionPlan",
    "PreservationRule",
    "PipelineConfig",
    # Recovery
    "RecoveryManifest",
]

__version__ = "0.1.0"
