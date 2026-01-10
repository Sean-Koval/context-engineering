"""Tests for graph node models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, NodeType, Role


class TestContent:
    """Tests for Content model."""

    def test_empty_content(self) -> None:
        """Content can be created with no fields."""
        content = Content()
        assert content.text is None
        assert content.role is None

    def test_message_content(self) -> None:
        """Content can hold message data."""
        content = Content(text="Hello!", role=Role.USER)
        assert content.text == "Hello!"
        assert content.role == Role.USER

    def test_tool_call_content(self) -> None:
        """Content can hold tool call data."""
        content = Content(
            tool_name="search",
            tool_args={"query": "test", "limit": 10},
        )
        assert content.tool_name == "search"
        assert content.tool_args == {"query": "test", "limit": 10}

    def test_tool_result_content(self) -> None:
        """Content can hold tool result data."""
        content = Content(
            tool_output={"results": [1, 2, 3]},
            is_error=False,
        )
        assert content.tool_output == {"results": [1, 2, 3]}
        assert content.is_error is False

    def test_tool_result_error(self) -> None:
        """Content can indicate tool errors."""
        content = Content(
            tool_output="Connection timeout",
            is_error=True,
        )
        assert content.is_error is True

    def test_artifact_content(self) -> None:
        """Content can hold artifact data."""
        content = Content(
            artifact_type="code",
            artifact_data="def hello(): pass",
            language="python",
        )
        assert content.artifact_type == "code"
        assert content.language == "python"

    def test_summary_content(self) -> None:
        """Content can hold summary data."""
        ids = [UUID("12345678-1234-1234-1234-123456789abc")]
        content = Content(
            text="Summary of previous messages",
            summarized_node_ids=ids,
            summary_method="hierarchical",
        )
        assert content.summarized_node_ids == ids
        assert content.summary_method == "hierarchical"

    def test_compression_tracking(self) -> None:
        """Content can track compression metadata."""
        content = Content(
            text="Compressed",
            original_tokens=1000,
            compressed_tokens=250,
        )
        assert content.original_tokens == 1000
        assert content.compressed_tokens == 250

    def test_external_ref(self) -> None:
        """Content can store external reference for evicted content."""
        content = Content(
            external_ref="s3://bucket/path/to/content.json",
        )
        assert content.external_ref == "s3://bucket/path/to/content.json"

    def test_forbids_extra_fields(self) -> None:
        """Content rejects unknown fields."""
        with pytest.raises(ValidationError):
            Content(unknown_field="value")


class TestNodeMetadata:
    """Tests for NodeMetadata model."""

    def test_default_values(self) -> None:
        """NodeMetadata has sensible defaults."""
        meta = NodeMetadata()
        assert meta.importance == 0.5
        assert meta.recency_score == 1.0
        assert meta.reference_count == 0
        assert meta.tags == set()
        assert meta.entities == []
        assert meta.pinned is False
        assert meta.access_count == 0

    def test_importance_bounds(self) -> None:
        """Importance must be between 0 and 1."""
        NodeMetadata(importance=0.0)  # Should work
        NodeMetadata(importance=1.0)  # Should work

        with pytest.raises(ValidationError):
            NodeMetadata(importance=-0.1)
        with pytest.raises(ValidationError):
            NodeMetadata(importance=1.1)

    def test_recency_bounds(self) -> None:
        """Recency score must be between 0 and 1."""
        NodeMetadata(recency_score=0.0)
        NodeMetadata(recency_score=1.0)

        with pytest.raises(ValidationError):
            NodeMetadata(recency_score=-0.1)
        with pytest.raises(ValidationError):
            NodeMetadata(recency_score=1.1)

    def test_reference_count_non_negative(self) -> None:
        """Reference count cannot be negative."""
        NodeMetadata(reference_count=0)
        NodeMetadata(reference_count=100)

        with pytest.raises(ValidationError):
            NodeMetadata(reference_count=-1)

    def test_tags_as_set(self) -> None:
        """Tags are stored as a set."""
        meta = NodeMetadata(tags={"tag1", "tag2"})
        assert meta.tags == {"tag1", "tag2"}

    def test_timestamps_auto_set(self) -> None:
        """Timestamps are automatically set."""
        before = datetime.now(UTC)
        meta = NodeMetadata()
        after = datetime.now(UTC)

        assert before <= meta.created_at <= after
        assert before <= meta.accessed_at <= after

    def test_preservation_flags(self) -> None:
        """Preservation flags work correctly."""
        future = datetime.now(UTC) + timedelta(hours=1)
        meta = NodeMetadata(
            pinned=True,
            preserve_until=future,
            min_compression_level=CompressionLevel.COMPACTED,
        )
        assert meta.pinned is True
        assert meta.preserve_until == future
        assert meta.min_compression_level == CompressionLevel.COMPACTED

    def test_provenance_fields(self) -> None:
        """Source tracking fields work."""
        meta = NodeMetadata(
            source_session="session-123",
            source_agent="agent-456",
        )
        assert meta.source_session == "session-123"
        assert meta.source_agent == "agent-456"

    def test_allows_extra_fields(self) -> None:
        """NodeMetadata allows custom fields via extra='allow'."""
        meta = NodeMetadata(custom_field="value")
        assert meta.custom_field == "value"


class TestContextNode:
    """Tests for ContextNode model."""

    def test_message_node(self) -> None:
        """Can create a message node."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hello", role=Role.USER),
        )
        assert node.type == NodeType.MESSAGE
        assert node.content.text == "Hello"
        assert isinstance(node.id, UUID)

    def test_tool_call_node(self) -> None:
        """Can create a tool call node."""
        node = ContextNode(
            type=NodeType.TOOL_CALL,
            content=Content(tool_name="search", tool_args={"q": "test"}),
        )
        assert node.type == NodeType.TOOL_CALL
        assert node.content.tool_name == "search"

    def test_default_compression_level(self) -> None:
        """Nodes start at FULL compression."""
        node = ContextNode(type=NodeType.MESSAGE, content=Content(text="Hi"))
        assert node.compression_level == CompressionLevel.FULL

    def test_token_count_optional(self) -> None:
        """Token count is optional."""
        node = ContextNode(type=NodeType.MESSAGE, content=Content(text="Hi"))
        assert node.token_count is None

        node.token_count = 5
        assert node.token_count == 5

    def test_sequence_number_optional(self) -> None:
        """Sequence number is set by ContextGraph."""
        node = ContextNode(type=NodeType.MESSAGE, content=Content(text="Hi"))
        assert node.sequence_number is None

    def test_compute_importance(self) -> None:
        """Importance calculation uses correct formula."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hi", role=Role.USER),
            metadata=NodeMetadata(
                importance=1.0,
                recency_score=1.0,
                reference_count=10,  # Capped at 1.0 (10/10)
            ),
        )
        # Formula: 0.4 * base + 0.3 * recency + 0.2 * refs + 0.1 * type_weight
        # MESSAGE type_weight = 0.8
        expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0 + 0.1 * 0.8
        assert node.compute_importance() == pytest.approx(expected)

    def test_compute_importance_low_refs(self) -> None:
        """Reference count is normalized to 0-1 range."""
        node = ContextNode(
            type=NodeType.SYSTEM,  # type_weight = 1.0
            content=Content(text="System prompt"),
            metadata=NodeMetadata(
                importance=0.5,
                recency_score=0.5,
                reference_count=5,  # Becomes 0.5 (5/10)
            ),
        )
        expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.2 * 0.5 + 0.1 * 1.0
        assert node.compute_importance() == pytest.approx(expected)

    def test_to_message_dict_user(self) -> None:
        """Message node converts to OpenAI format."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hello!", role=Role.USER),
        )
        msg = node.to_message_dict()
        assert msg == {"role": "user", "content": "Hello!"}

    def test_to_message_dict_assistant(self) -> None:
        """Assistant message converts correctly."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hi there!", role=Role.ASSISTANT),
        )
        msg = node.to_message_dict()
        assert msg == {"role": "assistant", "content": "Hi there!"}

    def test_to_message_dict_system(self) -> None:
        """System node converts correctly."""
        node = ContextNode(
            type=NodeType.SYSTEM,
            content=Content(text="You are helpful."),
        )
        msg = node.to_message_dict()
        assert msg == {"role": "system", "content": "You are helpful."}

    def test_to_message_dict_tool_call(self) -> None:
        """Tool call node converts to function call format."""
        node = ContextNode(
            type=NodeType.TOOL_CALL,
            content=Content(tool_name="search", tool_args={"q": "test"}),
        )
        msg = node.to_message_dict()
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"q": "test"}'

    def test_to_message_dict_tool_result(self) -> None:
        """Tool result node converts correctly."""
        node = ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(tool_output={"result": 42}),
        )
        msg = node.to_message_dict()
        assert msg["role"] == "tool"
        assert msg["content"] == '{"result": 42}'

    def test_mark_accessed(self) -> None:
        """mark_accessed updates metadata."""
        node = ContextNode(type=NodeType.MESSAGE, content=Content(text="Hi"))
        old_accessed = node.metadata.accessed_at
        old_count = node.metadata.access_count

        # Small delay to ensure timestamp changes
        import time

        time.sleep(0.001)

        node.mark_accessed()

        assert node.metadata.access_count == old_count + 1
        assert node.metadata.accessed_at > old_accessed
