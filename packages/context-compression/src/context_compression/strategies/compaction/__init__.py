"""Compaction compression strategies.

Compaction strategies reduce content while preserving key information.
They are mostly reversible - structure is known but some data may be lost.

Strategies:
- SchemaCompression: Extract JSON schemas from repeated structures
- EntityCentricCompression: Preserve entity context
- TaskRelevanceCompression: Remove off-task content (to be implemented)
"""

from __future__ import annotations

from context_compression.strategies.compaction.entity_centric import (
    EntityCentricCompression,
)
from context_compression.strategies.compaction.schema import SchemaCompression

__all__: list[str] = [
    "EntityCentricCompression",
    "SchemaCompression",
]
