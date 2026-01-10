"""Tests for graph type definitions."""

from __future__ import annotations

from context_core.graph.types import CompressionLevel, EdgeType, NodeType, Role


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_values_exist(self) -> None:
        """All expected node types exist."""
        assert NodeType.MESSAGE.value == "message"
        assert NodeType.TOOL_CALL.value == "tool_call"
        assert NodeType.TOOL_RESULT.value == "tool_result"
        assert NodeType.ARTIFACT.value == "artifact"
        assert NodeType.ENTITY.value == "entity"
        assert NodeType.SUMMARY.value == "summary"
        assert NodeType.SYSTEM.value == "system"
        assert NodeType.MEMORY.value == "memory"

    def test_is_string_enum(self) -> None:
        """NodeType can be used as string via .value."""
        assert NodeType.MESSAGE.value == "message"
        # str(Enum) subclass comparison works
        assert NodeType.MESSAGE == "message"

    def test_json_serializable(self) -> None:
        """NodeType serializes to JSON correctly."""
        import json

        data = {"type": NodeType.MESSAGE}
        assert json.dumps(data) == '{"type": "message"}'


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_values_exist(self) -> None:
        """All expected edge types exist."""
        assert EdgeType.TEMPORAL.value == "temporal"
        assert EdgeType.CAUSAL.value == "causal"
        assert EdgeType.REFERENCES.value == "references"
        assert EdgeType.SUMMARIZES.value == "summarizes"
        assert EdgeType.CONTRADICTS.value == "contradicts"
        assert EdgeType.DEPENDS_ON.value == "depends_on"
        assert EdgeType.SAME_ENTITY.value == "same_entity"
        assert EdgeType.PARENT_CHILD.value == "parent_child"
        assert EdgeType.TOOL_IO.value == "tool_io"


class TestCompressionLevel:
    """Tests for CompressionLevel enum."""

    def test_ordering(self) -> None:
        """Compression levels are ordered correctly."""
        assert CompressionLevel.FULL < CompressionLevel.COMPACTED
        assert CompressionLevel.COMPACTED < CompressionLevel.SUMMARIZED
        assert CompressionLevel.SUMMARIZED < CompressionLevel.EVICTED

    def test_int_values(self) -> None:
        """Compression levels have correct integer values."""
        assert CompressionLevel.FULL.value == 0
        assert CompressionLevel.COMPACTED.value == 1
        assert CompressionLevel.SUMMARIZED.value == 2
        assert CompressionLevel.EVICTED.value == 3

    def test_comparison(self) -> None:
        """Can compare compression levels."""
        assert CompressionLevel.COMPACTED <= CompressionLevel.SUMMARIZED
        assert CompressionLevel.FULL < CompressionLevel.EVICTED


class TestRole:
    """Tests for Role enum."""

    def test_all_values_exist(self) -> None:
        """All expected roles exist."""
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.SYSTEM.value == "system"
        assert Role.TOOL.value == "tool"
