"""Edge models for the context graph.

This module defines the edge types:
- EdgeMetadata: Metadata for graph edges
- Edge: A directed edge between two nodes
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from context_core.graph.types import EdgeType


class EdgeMetadata(BaseModel):
    """Metadata for graph edges.

    Attributes:
        weight: Edge weight for graph algorithms (default 1.0)
        created_at: When the edge was created
        properties: Additional custom properties
    """

    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    properties: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    """A directed edge between two nodes in the context graph.

    Edges represent relationships between nodes such as:
    - TEMPORAL: Sequential ordering (A happened before B)
    - CAUSAL: Causation (A caused B)
    - REFERENCES: A mentions or uses B
    - TOOL_IO: Links tool_call to its tool_result

    Attributes:
        id: Unique identifier for this edge
        source_id: ID of the source node
        target_id: ID of the target node
        type: The type of relationship
        metadata: Edge metadata including weight and properties
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    type: EdgeType
    metadata: EdgeMetadata = Field(default_factory=EdgeMetadata)
