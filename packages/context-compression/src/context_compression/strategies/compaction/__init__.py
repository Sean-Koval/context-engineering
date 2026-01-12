"""Compaction compression strategies.

Compaction strategies reduce content while preserving key information.
They are mostly reversible - structure is known but some data may be lost.

Strategies:
- SchemaCompression: Extract JSON schemas from repeated structures
- TaskRelevanceCompression: Compress off-task content based on semantic similarity
- EntityCentricCompression: Preserve entity context (to be implemented)
"""

from __future__ import annotations

from context_compression.strategies.compaction.schema import SchemaCompression
from context_compression.strategies.compaction.task import TaskRelevanceCompression

__all__: list[str] = [
    "SchemaCompression",
    "TaskRelevanceCompression",
]
