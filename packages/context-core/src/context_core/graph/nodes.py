"""Node models for the context graph.

This module defines the core node types:
- Content: Polymorphic content container for different node types
- NodeMetadata: Metadata for importance scoring and lifecycle
- ContextNode: A node in the context graph
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from context_core.graph.types import CompressionLevel, NodeType, Role


class Content(BaseModel):
    """Polymorphic content container for different node types.

    This model uses optional fields for each node type rather than
    discriminated unions, allowing flexibility while maintaining
    type safety through validation.

    Attributes:
        text: Common text content (used by MESSAGE, SYSTEM, etc.)
        role: Message role (MESSAGE nodes)
        tool_name: Name of the tool (TOOL_CALL nodes)
        tool_args: Arguments passed to tool (TOOL_CALL nodes)
        tool_output: Result from tool execution (TOOL_RESULT nodes)
        is_error: Whether tool execution failed (TOOL_RESULT nodes)
        artifact_type: Type of artifact - "code", "file", "json" (ARTIFACT nodes)
        artifact_data: The artifact content (ARTIFACT nodes)
        language: Programming language (code ARTIFACT nodes)
        file_path: Path to file (file ARTIFACT nodes)
        summarized_node_ids: IDs of nodes this summarizes (SUMMARY nodes)
        summary_method: Method used - "hierarchical", "task_aware" (SUMMARY nodes)
        memory_key: Key for memory retrieval (MEMORY nodes)
        retrieval_score: Relevance score from retrieval (MEMORY nodes)
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        external_ref: URI for externalized content (EVICTED nodes)
    """

    model_config = ConfigDict(extra="forbid")

    # Common fields
    text: str | None = None

    # MESSAGE-specific
    role: Role | None = None

    # TOOL_CALL-specific
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    # TOOL_RESULT-specific
    tool_output: Any | None = None
    is_error: bool = False

    # ARTIFACT-specific
    artifact_type: str | None = None  # "code", "file", "json", etc.
    artifact_data: Any | None = None
    language: str | None = None  # For code artifacts
    file_path: str | None = None  # For file artifacts

    # SUMMARY-specific
    summarized_node_ids: list[UUID] | None = None
    summary_method: str | None = None  # "hierarchical", "task_aware", etc.

    # MEMORY-specific
    memory_key: str | None = None
    retrieval_score: float | None = None

    # Compression tracking
    original_tokens: int | None = None
    compressed_tokens: int | None = None
    external_ref: str | None = None  # URI for externalized content


class NodeMetadata(BaseModel):
    """Metadata for importance scoring, filtering, and lifecycle management.

    Attributes:
        importance: Base importance score (0.0 - 1.0)
        recency_score: How recent the node is (0.0 - 1.0, decays over time)
        reference_count: How many other nodes reference this one
        tags: Classification tags for filtering
        entities: IDs of entities mentioned in this node
        created_at: When the node was created
        accessed_at: When the node was last accessed
        access_count: Number of times the node has been accessed
        pinned: If True, never compress this node
        preserve_until: Don't compress until this datetime
        min_compression_level: Minimum compression level to apply
        source_session: ID of the session that created this node
        source_agent: ID of the agent that created this node
    """

    model_config = ConfigDict(extra="allow")  # Allow custom metadata fields

    # Importance scoring (0.0 - 1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reference_count: int = Field(default=0, ge=0)

    # Classification
    tags: set[str] = Field(default_factory=set)
    entities: list[str] = Field(default_factory=list)  # Entity IDs

    # Lifecycle
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    accessed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = Field(default=0, ge=0)

    # Preservation flags
    pinned: bool = False  # Never compress
    preserve_until: datetime | None = None
    min_compression_level: CompressionLevel = CompressionLevel.FULL

    # Provenance
    source_session: str | None = None
    source_agent: str | None = None


# Type weights for importance calculation
_TYPE_WEIGHTS: dict[NodeType, float] = {
    NodeType.SYSTEM: 1.0,
    NodeType.MESSAGE: 0.8,
    NodeType.TOOL_RESULT: 0.7,
    NodeType.TOOL_CALL: 0.6,
    NodeType.ARTIFACT: 0.7,
    NodeType.ENTITY: 0.5,
    NodeType.SUMMARY: 0.6,
    NodeType.MEMORY: 0.5,
}


class ContextNode(BaseModel):
    """A node in the context graph representing a unit of context.

    Each node has a type, content appropriate for that type, metadata
    for importance scoring and lifecycle, and compression state.

    Attributes:
        id: Unique identifier for this node
        type: The type of node (MESSAGE, TOOL_CALL, etc.)
        content: The node's content
        metadata: Metadata for scoring and lifecycle
        compression_level: Current compression state
        token_count: Number of tokens in this node's content
        sequence_number: Position in temporal ordering (set by ContextGraph)
    """

    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    type: NodeType
    content: Content
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    # Compression state
    compression_level: CompressionLevel = CompressionLevel.FULL

    # Token tracking
    token_count: int | None = None

    # Graph position (set by ContextGraph)
    sequence_number: int | None = None

    def compute_importance(self) -> float:
        """Calculate composite importance score.

        Formula: 0.4 * base + 0.3 * recency + 0.2 * refs + 0.1 * type_weight

        Returns:
            Composite importance score between 0.0 and 1.0
        """
        base = self.metadata.importance
        recency = self.metadata.recency_score
        refs = min(self.metadata.reference_count / 10, 1.0)
        type_weight = _TYPE_WEIGHTS.get(self.type, 0.5)

        return 0.4 * base + 0.3 * recency + 0.2 * refs + 0.1 * type_weight

    def to_message_dict(self) -> dict[str, Any]:
        """Convert to LLM message format (OpenAI-style).

        Returns:
            Dictionary in OpenAI chat completion message format.
        """
        if self.type == NodeType.MESSAGE:
            return {
                "role": self.content.role.value if self.content.role else "user",
                "content": self.content.text or "",
            }
        elif self.type == NodeType.TOOL_CALL:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(self.id),
                        "type": "function",
                        "function": {
                            "name": self.content.tool_name,
                            "arguments": json.dumps(self.content.tool_args or {}),
                        },
                    }
                ],
            }
        elif self.type == NodeType.TOOL_RESULT:
            content = (
                json.dumps(self.content.tool_output)
                if self.content.tool_output is not None
                else ""
            )
            return {
                "role": "tool",
                "tool_call_id": str(self.id),
                "content": content,
            }
        elif self.type == NodeType.SYSTEM:
            return {
                "role": "system",
                "content": self.content.text or "",
            }
        else:
            # For other types (ARTIFACT, ENTITY, SUMMARY, MEMORY),
            # render as assistant message
            content = self.content.text or str(self.content.artifact_data or "")
            return {
                "role": "assistant",
                "content": content,
            }

    def mark_accessed(self) -> None:
        """Update access tracking metadata."""
        self.metadata.accessed_at = datetime.now(UTC)
        self.metadata.access_count += 1
