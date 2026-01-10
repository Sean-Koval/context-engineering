"""Graph module - Graph-based context representation.

Components:
- ContextNode: Individual context items (messages, tool calls, etc.)
- Edge: Relationships between nodes
- ContextGraph: Main graph structure with CRUD operations
"""

from __future__ import annotations

from context_core.graph.types import (
    CompressionLevel,
    EdgeType,
    NodeType,
    Role,
)

__all__ = [
    "CompressionLevel",
    "EdgeType",
    "NodeType",
    "Role",
]
