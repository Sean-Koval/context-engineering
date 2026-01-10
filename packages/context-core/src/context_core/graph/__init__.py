"""Graph module - Graph-based context representation.

Components:
- ContextNode: Individual context items (messages, tool calls, etc.)
- Edge: Relationships between nodes
- ContextGraph: Main graph structure with CRUD operations
- GraphStats: Statistics about the context graph
"""

from __future__ import annotations

from context_core.graph.context_graph import ContextGraph, GraphStats
from context_core.graph.edges import Edge, EdgeMetadata
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import (
    CompressionLevel,
    EdgeType,
    NodeType,
    Role,
)

__all__ = [
    # Types
    "CompressionLevel",
    "EdgeType",
    "NodeType",
    "Role",
    # Nodes
    "Content",
    "ContextNode",
    "NodeMetadata",
    # Edges
    "Edge",
    "EdgeMetadata",
    # Graph
    "ContextGraph",
    "GraphStats",
]
