"""Compaction compression strategies.

Compaction strategies reduce content while preserving key information.
They are mostly reversible - structure is known but some data may be lost.

Strategies (to be implemented):
- SchemaCompression: Extract JSON schemas
- EntityCentricCompression: Preserve entity context
- TaskRelevanceCompression: Remove off-task content
"""

from __future__ import annotations

__all__: list[str] = []
