"""Tests for graph edge models."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from context_core.graph.edges import Edge, EdgeMetadata
from context_core.graph.types import EdgeType


class TestEdgeMetadata:
    """Tests for EdgeMetadata model."""

    def test_default_values(self) -> None:
        """EdgeMetadata has sensible defaults."""
        meta = EdgeMetadata()
        assert meta.weight == 1.0
        assert meta.properties == {}
        assert isinstance(meta.created_at, datetime)

    def test_weight_non_negative(self) -> None:
        """Weight must be non-negative."""
        EdgeMetadata(weight=0.0)
        EdgeMetadata(weight=100.0)

        with pytest.raises(ValidationError):
            EdgeMetadata(weight=-0.1)

    def test_custom_properties(self) -> None:
        """Can store custom properties."""
        meta = EdgeMetadata(
            properties={
                "confidence": 0.95,
                "source": "auto-detected",
            }
        )
        assert meta.properties["confidence"] == 0.95
        assert meta.properties["source"] == "auto-detected"

    def test_created_at_auto_set(self) -> None:
        """created_at is automatically set."""
        before = datetime.now(UTC)
        meta = EdgeMetadata()
        after = datetime.now(UTC)

        assert before <= meta.created_at <= after


class TestEdge:
    """Tests for Edge model."""

    def test_create_edge(self) -> None:
        """Can create an edge between two nodes."""
        source = uuid4()
        target = uuid4()

        edge = Edge(
            source_id=source,
            target_id=target,
            type=EdgeType.TEMPORAL,
        )

        assert edge.source_id == source
        assert edge.target_id == target
        assert edge.type == EdgeType.TEMPORAL
        assert isinstance(edge.id, UUID)

    def test_edge_with_metadata(self) -> None:
        """Edge can have custom metadata."""
        edge = Edge(
            source_id=uuid4(),
            target_id=uuid4(),
            type=EdgeType.REFERENCES,
            metadata=EdgeMetadata(weight=0.5),
        )

        assert edge.metadata.weight == 0.5

    def test_default_metadata(self) -> None:
        """Edge gets default metadata if not specified."""
        edge = Edge(
            source_id=uuid4(),
            target_id=uuid4(),
            type=EdgeType.CAUSAL,
        )

        assert edge.metadata.weight == 1.0

    def test_all_edge_types(self) -> None:
        """Can create edges of all types."""
        source = uuid4()
        target = uuid4()

        for edge_type in EdgeType:
            edge = Edge(source_id=source, target_id=target, type=edge_type)
            assert edge.type == edge_type

    def test_tool_io_edge(self) -> None:
        """TOOL_IO edge links tool call to result."""
        call_id = uuid4()
        result_id = uuid4()

        edge = Edge(
            source_id=call_id,
            target_id=result_id,
            type=EdgeType.TOOL_IO,
        )

        assert edge.type == EdgeType.TOOL_IO

    def test_summarizes_edge(self) -> None:
        """SUMMARIZES edge for summary nodes."""
        summary_id = uuid4()
        original_id = uuid4()

        edge = Edge(
            source_id=summary_id,
            target_id=original_id,
            type=EdgeType.SUMMARIZES,
        )

        assert edge.type == EdgeType.SUMMARIZES
