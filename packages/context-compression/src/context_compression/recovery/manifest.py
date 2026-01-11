"""Recovery manifest for tracking compression operations.

The RecoveryManifest maintains a log of all compression operations
performed during a session, enabling potential recovery of compressed content.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from context_compression.recovery.operations import (
    CollapseOperation,
    CompactOperation,
    CompressionOperation,
    DeduplicateOperation,
    EvictOperation,
    ExternalizeOperation,
    SummarizeOperation,
)


class ManifestStats(BaseModel):
    """Statistics about compression operations in a manifest.

    Attributes:
        total_operations: Total number of operations logged
        tokens_saved: Total tokens saved across all operations
        nodes_affected: Number of unique nodes affected
        recoverable_operations: Number of operations that can be reversed
        operations_by_type: Count of operations by type
    """

    total_operations: int = 0
    tokens_saved: int = 0
    nodes_affected: int = 0
    recoverable_operations: int = 0
    operations_by_type: dict[str, int] = Field(default_factory=dict)


class RecoveryManifest(BaseModel):
    """Manifest tracking all compression operations for recovery.

    The manifest logs every compression operation with enough information
    to potentially reverse it. It provides methods for:
    - Logging new operations
    - Querying recovery capability
    - Computing statistics
    - Serialization for persistence

    Attributes:
        id: Unique identifier for this manifest
        session_id: ID of the session this manifest belongs to
        created_at: When the manifest was created
        enable_recovery: Whether to store recovery data
        operations: List of all compression operations
    """

    id: UUID = Field(default_factory=uuid4)
    session_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    enable_recovery: bool = True
    operations: list[CompressionOperation] = Field(default_factory=list)

    # Index for fast lookup by node_id
    _node_index: dict[UUID, list[int]] = {}

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Build node index after initialization."""
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the node-to-operations index."""
        self._node_index = {}
        for i, op in enumerate(self.operations):
            self._node_index.setdefault(op.node_id, []).append(i)

    def log_operation(self, operation: CompressionOperation) -> None:
        """Log a compression operation.

        Args:
            operation: The operation to log
        """
        idx = len(self.operations)
        self.operations.append(operation)
        self._node_index.setdefault(operation.node_id, []).append(idx)

    def log_externalize(
        self,
        node_id: UUID,
        external_uri: str,
        original_tokens: int,
        content_hash: str,
        preview: str | None = None,
        storage_backend: str = "memory",
    ) -> ExternalizeOperation:
        """Log an externalization operation.

        Args:
            node_id: ID of the node being externalized
            external_uri: URI where content is stored
            original_tokens: Token count before externalization
            content_hash: Hash of original content
            preview: Short preview of the content
            storage_backend: Name of the storage backend

        Returns:
            The created operation
        """
        op = ExternalizeOperation(
            node_id=node_id,
            external_uri=external_uri,
            original_tokens=original_tokens,
            original_content_hash=content_hash,
            preview=preview,
            storage_backend=storage_backend,
        )
        self.log_operation(op)
        return op

    def log_deduplicate(
        self,
        node_id: UUID,
        removed_node_ids: list[UUID],
        kept_node_id: UUID,
        original_tokens: int,
        similarity_score: float,
        original_contents: dict[str, str] | None = None,
    ) -> DeduplicateOperation:
        """Log a deduplication operation.

        Args:
            node_id: Primary node affected
            removed_node_ids: IDs of removed duplicates
            kept_node_id: ID of the kept canonical node
            original_tokens: Token count of removed nodes
            similarity_score: How similar the nodes were
            original_contents: Serialized original contents

        Returns:
            The created operation
        """
        op = DeduplicateOperation(
            node_id=node_id,
            removed_node_ids=removed_node_ids,
            kept_node_id=kept_node_id,
            original_tokens=original_tokens,
            similarity_score=similarity_score,
            original_contents=original_contents or {},
        )
        self.log_operation(op)
        return op

    def log_collapse(
        self,
        node_id: UUID,
        original_node_ids: list[UUID],
        collapsed_node_id: UUID,
        original_tokens: int,
        chain_description: str = "",
        original_sequence: list[dict] | None = None,
    ) -> CollapseOperation:
        """Log a collapse operation.

        Args:
            node_id: Primary node affected
            original_node_ids: IDs of nodes in the chain
            collapsed_node_id: ID of the summary node
            original_tokens: Token count of original chain
            chain_description: Description of the chain
            original_sequence: Serialized original sequence

        Returns:
            The created operation
        """
        op = CollapseOperation(
            node_id=node_id,
            original_node_ids=original_node_ids,
            collapsed_node_id=collapsed_node_id,
            original_tokens=original_tokens,
            chain_description=chain_description,
            original_sequence=original_sequence or [],
        )
        self.log_operation(op)
        return op

    def log_summarize(
        self,
        node_id: UUID,
        original_node_ids: list[UUID],
        summary_node_id: UUID,
        original_tokens: int,
        summary_tokens: int,
        method: str = "hierarchical",
        summary_text: str = "",
    ) -> SummarizeOperation:
        """Log a summarization operation.

        Args:
            node_id: Primary node affected
            original_node_ids: IDs of summarized nodes
            summary_node_id: ID of the summary node
            original_tokens: Token count before summarization
            summary_tokens: Token count of summary
            method: Summarization method used
            summary_text: The generated summary

        Returns:
            The created operation
        """
        op = SummarizeOperation(
            node_id=node_id,
            original_node_ids=original_node_ids,
            summary_node_id=summary_node_id,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            method=method,
            summary_text=summary_text,
        )
        self.log_operation(op)
        return op

    def log_evict(
        self,
        node_id: UUID,
        original_tokens: int,
        reason: str = "low_importance",
        external_ref: str | None = None,
    ) -> EvictOperation:
        """Log an eviction operation.

        Args:
            node_id: ID of the evicted node
            original_tokens: Token count before eviction
            reason: Why the node was evicted
            external_ref: External reference if saved

        Returns:
            The created operation
        """
        op = EvictOperation(
            node_id=node_id,
            original_tokens=original_tokens,
            reason=reason,
            external_ref=external_ref,
        )
        self.log_operation(op)
        return op

    def get_operations_for_node(self, node_id: UUID) -> list[CompressionOperation]:
        """Get all operations affecting a specific node.

        Args:
            node_id: The node ID to look up

        Returns:
            List of operations affecting this node
        """
        indices = self._node_index.get(node_id, [])
        return [self.operations[i] for i in indices]

    def can_recover_node(self, node_id: UUID) -> bool:
        """Check if a node's original content can be recovered.

        A node is recoverable if ALL operations affecting it are recoverable.

        Args:
            node_id: The node ID to check

        Returns:
            True if the node can be fully recovered
        """
        ops = self.get_operations_for_node(node_id)
        if not ops:
            return True  # No operations = nothing to recover
        return all(op.is_recoverable for op in ops)

    def get_recoverable_nodes(self) -> set[UUID]:
        """Get IDs of all nodes that can be recovered.

        Returns:
            Set of node IDs that can be recovered
        """
        return {nid for nid in self._node_index if self.can_recover_node(nid)}

    def get_stats(self) -> ManifestStats:
        """Compute statistics about the manifest.

        Returns:
            ManifestStats with operation counts and totals
        """
        stats = ManifestStats()
        stats.total_operations = len(self.operations)
        stats.nodes_affected = len(self._node_index)

        for op in self.operations:
            # Count by type
            op_type = op.op_type
            stats.operations_by_type[op_type] = (
                stats.operations_by_type.get(op_type, 0) + 1
            )

            # Count tokens saved
            if hasattr(op, "compressed_tokens"):
                stats.tokens_saved += op.original_tokens - op.compressed_tokens
            elif hasattr(op, "summary_tokens"):
                stats.tokens_saved += op.original_tokens - op.summary_tokens
            elif op.op_type == "externalize":
                # Externalization saves most tokens (keep just preview)
                stats.tokens_saved += int(op.original_tokens * 0.9)
            elif op.op_type == "evict":
                stats.tokens_saved += op.original_tokens

            # Count recoverable
            if op.is_recoverable:
                stats.recoverable_operations += 1

        return stats

    def clear(self) -> None:
        """Clear all operations from the manifest."""
        self.operations.clear()
        self._node_index.clear()

    def to_dict(self) -> dict:
        """Serialize manifest to a dictionary.

        Returns:
            Dictionary representation for persistence
        """
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "enable_recovery": self.enable_recovery,
            "operations": [op.model_dump(mode="json") for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryManifest:
        """Deserialize manifest from a dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Reconstructed RecoveryManifest
        """
        # Map op_type to operation class
        op_classes = {
            "externalize": ExternalizeOperation,
            "deduplicate": DeduplicateOperation,
            "collapse": CollapseOperation,
            "compact": CompactOperation,
            "summarize": SummarizeOperation,
            "evict": EvictOperation,
        }

        operations = []
        for op_data in data.get("operations", []):
            op_type = op_data.get("op_type")
            if op_type in op_classes:
                operations.append(op_classes[op_type].model_validate(op_data))

        return cls(
            id=UUID(data["id"]),
            session_id=data.get("session_id", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            enable_recovery=data.get("enable_recovery", True),
            operations=operations,
        )
