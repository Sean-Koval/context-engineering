"""Operation types for compression recovery.

Each operation type represents a specific kind of compression action
that can be logged to the RecoveryManifest. Operations are categorized
by their recoverability:

- Fully recoverable: ExternalizeOperation, DeduplicateOperation, CollapseOperation
- Partially recoverable: CompactOperation
- Not recoverable: SummarizeOperation, EvictOperation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseOperation(BaseModel):
    """Base class for all compression operations.

    Attributes:
        id: Unique identifier for this operation
        timestamp: When the operation occurred
        node_id: Primary node affected by the operation
        original_tokens: Token count before operation
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node_id: UUID
    original_tokens: int = Field(ge=0)


class ExternalizeOperation(BaseOperation):
    """Tracks externalized content for recovery.

    When content is externalized, the original data is stored externally
    and a URI reference is kept in the node. This is fully reversible
    by fetching the content from the external storage.

    Attributes:
        op_type: Discriminator for operation type
        external_uri: URI where content is stored
        original_content_hash: Hash of original content for verification
        preview: Short preview of the externalized content
        storage_backend: Name of the storage backend used
    """

    op_type: Literal["externalize"] = "externalize"
    external_uri: str
    original_content_hash: str
    preview: str | None = None
    storage_backend: str = "memory"

    @property
    def is_recoverable(self) -> bool:
        """Externalized content can always be recovered."""
        return True


class DeduplicateOperation(BaseOperation):
    """Tracks deduplicated nodes.

    When semantically similar nodes are found, duplicates are removed
    and only one canonical node is kept. This is recoverable by
    restoring the removed nodes.

    Attributes:
        op_type: Discriminator for operation type
        removed_node_ids: IDs of nodes that were removed
        kept_node_id: ID of the canonical node that was kept
        similarity_score: How similar the nodes were (0.0-1.0)
        original_contents: Serialized original contents (for recovery)
    """

    op_type: Literal["deduplicate"] = "deduplicate"
    removed_node_ids: list[UUID] = Field(default_factory=list)
    kept_node_id: UUID
    similarity_score: float = Field(ge=0.0, le=1.0)
    original_contents: dict[str, str] = Field(default_factory=dict)

    @property
    def is_recoverable(self) -> bool:
        """Deduplicated content is recoverable if originals were saved."""
        return bool(self.original_contents)


class CollapseOperation(BaseOperation):
    """Tracks collapsed tool chains.

    When a sequence of tool calls is collapsed into a summary,
    the original sequence can be restored from this operation.

    Attributes:
        op_type: Discriminator for operation type
        original_node_ids: IDs of nodes in the original chain
        collapsed_node_id: ID of the summary node
        chain_description: Human-readable description of the chain
        original_sequence: Serialized original sequence (for recovery)
    """

    op_type: Literal["collapse"] = "collapse"
    original_node_ids: list[UUID] = Field(default_factory=list)
    collapsed_node_id: UUID
    chain_description: str = ""
    original_sequence: list[dict] = Field(default_factory=list)

    @property
    def is_recoverable(self) -> bool:
        """Collapsed chains are recoverable if sequence was saved."""
        return bool(self.original_sequence)


class CompactOperation(BaseOperation):
    """Tracks compaction operations.

    Compaction reduces content while preserving key information.
    This includes schema compression, entity-centric compression, etc.

    Attributes:
        op_type: Discriminator for operation type
        compaction_method: Name of the compaction method used
        compressed_tokens: Token count after compaction
        preserved_fields: Fields/content that was preserved
        removed_fields: Fields/content that was removed
    """

    op_type: Literal["compact"] = "compact"
    compaction_method: str
    compressed_tokens: int = Field(ge=0)
    preserved_fields: list[str] = Field(default_factory=list)
    removed_fields: list[str] = Field(default_factory=list)

    @property
    def is_recoverable(self) -> bool:
        """Compaction is partially recoverable - structure known but data lost."""
        return False


class SummarizeOperation(BaseOperation):
    """Tracks summarization operations.

    Summarization creates a condensed representation of one or more nodes.
    This is not recoverable - the original detailed content is lost.

    Attributes:
        op_type: Discriminator for operation type
        original_node_ids: IDs of nodes that were summarized
        summary_node_id: ID of the created summary node
        summary_tokens: Token count of the summary
        method: Summarization method used
        summary_text: The generated summary text
    """

    op_type: Literal["summarize"] = "summarize"
    original_node_ids: list[UUID] = Field(default_factory=list)
    summary_node_id: UUID
    summary_tokens: int = Field(ge=0)
    method: str = "hierarchical"
    summary_text: str = ""

    @property
    def is_recoverable(self) -> bool:
        """Summarization is not recoverable - original content is lost."""
        return False


class EvictOperation(BaseOperation):
    """Tracks eviction operations.

    Eviction removes content entirely, keeping only a pointer.
    This is not recoverable unless the content was externalized first.

    Attributes:
        op_type: Discriminator for operation type
        reason: Why the node was evicted
        external_ref: External reference if content was saved externally
    """

    op_type: Literal["evict"] = "evict"
    reason: str = "low_importance"
    external_ref: str | None = None

    @property
    def is_recoverable(self) -> bool:
        """Eviction is only recoverable if externally stored."""
        return self.external_ref is not None


# Union type for all compression operations with discriminator
CompressionOperation = Annotated[
    ExternalizeOperation
    | DeduplicateOperation
    | CollapseOperation
    | CompactOperation
    | SummarizeOperation
    | EvictOperation,
    Field(discriminator="op_type"),
]
