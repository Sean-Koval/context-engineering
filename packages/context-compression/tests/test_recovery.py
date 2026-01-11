"""Tests for recovery manifest and operations."""

from __future__ import annotations

from uuid import uuid4

from context_compression.recovery import (
    CollapseOperation,
    CompactOperation,
    DeduplicateOperation,
    EvictOperation,
    ExternalizeOperation,
    RecoveryManifest,
    SummarizeOperation,
)


class TestExternalizeOperation:
    """Tests for ExternalizeOperation."""

    def test_create_operation(self):
        """Test creating an externalize operation."""
        node_id = uuid4()
        op = ExternalizeOperation(
            node_id=node_id,
            external_uri="memory://test",
            original_tokens=1000,
            original_content_hash="abc123",
            preview="Preview text...",
        )
        assert op.node_id == node_id
        assert op.external_uri == "memory://test"
        assert op.original_tokens == 1000
        assert op.original_content_hash == "abc123"
        assert op.preview == "Preview text..."
        assert op.op_type == "externalize"

    def test_is_recoverable(self):
        """Test externalize is always recoverable."""
        op = ExternalizeOperation(
            node_id=uuid4(),
            external_uri="memory://test",
            original_tokens=100,
            original_content_hash="hash",
        )
        assert op.is_recoverable is True


class TestDeduplicateOperation:
    """Tests for DeduplicateOperation."""

    def test_create_operation(self):
        """Test creating a deduplicate operation."""
        node_id = uuid4()
        removed = [uuid4(), uuid4()]
        kept = uuid4()

        op = DeduplicateOperation(
            node_id=node_id,
            removed_node_ids=removed,
            kept_node_id=kept,
            original_tokens=500,
            similarity_score=0.95,
        )
        assert op.node_id == node_id
        assert len(op.removed_node_ids) == 2
        assert op.kept_node_id == kept
        assert op.similarity_score == 0.95
        assert op.op_type == "deduplicate"

    def test_is_recoverable_with_contents(self):
        """Test deduplicate recoverable when contents saved."""
        op = DeduplicateOperation(
            node_id=uuid4(),
            removed_node_ids=[uuid4()],
            kept_node_id=uuid4(),
            original_tokens=100,
            similarity_score=0.9,
            original_contents={"id1": "content1"},
        )
        assert op.is_recoverable is True

    def test_not_recoverable_without_contents(self):
        """Test deduplicate not recoverable without contents."""
        op = DeduplicateOperation(
            node_id=uuid4(),
            removed_node_ids=[uuid4()],
            kept_node_id=uuid4(),
            original_tokens=100,
            similarity_score=0.9,
        )
        assert op.is_recoverable is False


class TestCollapseOperation:
    """Tests for CollapseOperation."""

    def test_create_operation(self):
        """Test creating a collapse operation."""
        node_id = uuid4()
        original_ids = [uuid4(), uuid4(), uuid4()]
        collapsed_id = uuid4()

        op = CollapseOperation(
            node_id=node_id,
            original_node_ids=original_ids,
            collapsed_node_id=collapsed_id,
            original_tokens=1500,
            chain_description="3 read_file calls",
        )
        assert op.node_id == node_id
        assert len(op.original_node_ids) == 3
        assert op.collapsed_node_id == collapsed_id
        assert op.chain_description == "3 read_file calls"
        assert op.op_type == "collapse"

    def test_is_recoverable_with_sequence(self):
        """Test collapse recoverable when sequence saved."""
        op = CollapseOperation(
            node_id=uuid4(),
            original_node_ids=[uuid4()],
            collapsed_node_id=uuid4(),
            original_tokens=100,
            original_sequence=[{"call": "data"}],
        )
        assert op.is_recoverable is True

    def test_not_recoverable_without_sequence(self):
        """Test collapse not recoverable without sequence."""
        op = CollapseOperation(
            node_id=uuid4(),
            original_node_ids=[uuid4()],
            collapsed_node_id=uuid4(),
            original_tokens=100,
        )
        assert op.is_recoverable is False


class TestCompactOperation:
    """Tests for CompactOperation."""

    def test_create_operation(self):
        """Test creating a compact operation."""
        op = CompactOperation(
            node_id=uuid4(),
            original_tokens=500,
            compaction_method="schema",
            compressed_tokens=100,
            preserved_fields=["id", "name"],
            removed_fields=["data", "metadata"],
        )
        assert op.compaction_method == "schema"
        assert op.compressed_tokens == 100
        assert "id" in op.preserved_fields
        assert "data" in op.removed_fields
        assert op.op_type == "compact"

    def test_not_recoverable(self):
        """Test compact is not recoverable."""
        op = CompactOperation(
            node_id=uuid4(),
            original_tokens=100,
            compaction_method="test",
            compressed_tokens=50,
        )
        assert op.is_recoverable is False


class TestSummarizeOperation:
    """Tests for SummarizeOperation."""

    def test_create_operation(self):
        """Test creating a summarize operation."""
        original_ids = [uuid4(), uuid4()]
        summary_id = uuid4()

        op = SummarizeOperation(
            node_id=uuid4(),
            original_node_ids=original_ids,
            summary_node_id=summary_id,
            original_tokens=2000,
            summary_tokens=200,
            method="hierarchical",
            summary_text="Summary of content...",
        )
        assert len(op.original_node_ids) == 2
        assert op.summary_node_id == summary_id
        assert op.summary_tokens == 200
        assert op.method == "hierarchical"
        assert op.op_type == "summarize"

    def test_not_recoverable(self):
        """Test summarize is not recoverable."""
        op = SummarizeOperation(
            node_id=uuid4(),
            original_node_ids=[uuid4()],
            summary_node_id=uuid4(),
            original_tokens=1000,
            summary_tokens=100,
        )
        assert op.is_recoverable is False


class TestEvictOperation:
    """Tests for EvictOperation."""

    def test_create_operation(self):
        """Test creating an evict operation."""
        op = EvictOperation(
            node_id=uuid4(),
            original_tokens=500,
            reason="low_importance",
        )
        assert op.reason == "low_importance"
        assert op.external_ref is None
        assert op.op_type == "evict"

    def test_recoverable_with_external_ref(self):
        """Test evict recoverable when externally stored."""
        op = EvictOperation(
            node_id=uuid4(),
            original_tokens=100,
            external_ref="s3://bucket/key",
        )
        assert op.is_recoverable is True

    def test_not_recoverable_without_external_ref(self):
        """Test evict not recoverable without external ref."""
        op = EvictOperation(
            node_id=uuid4(),
            original_tokens=100,
        )
        assert op.is_recoverable is False


class TestRecoveryManifest:
    """Tests for RecoveryManifest."""

    def test_create_manifest(self):
        """Test creating a manifest."""
        manifest = RecoveryManifest(session_id="test-session")
        assert manifest.session_id == "test-session"
        assert len(manifest.operations) == 0
        assert manifest.enable_recovery is True

    def test_log_externalize(self):
        """Test logging externalize operation."""
        manifest = RecoveryManifest()
        node_id = uuid4()

        op = manifest.log_externalize(
            node_id=node_id,
            external_uri="memory://test",
            original_tokens=1000,
            content_hash="abc123",
            preview="Preview...",
        )

        assert len(manifest.operations) == 1
        assert manifest.operations[0] == op
        assert op.node_id == node_id

    def test_log_deduplicate(self):
        """Test logging deduplicate operation."""
        manifest = RecoveryManifest()
        node_id = uuid4()
        kept_id = uuid4()

        op = manifest.log_deduplicate(
            node_id=node_id,
            removed_node_ids=[uuid4()],
            kept_node_id=kept_id,
            original_tokens=500,
            similarity_score=0.95,
        )

        assert len(manifest.operations) == 1
        assert op.kept_node_id == kept_id

    def test_log_collapse(self):
        """Test logging collapse operation."""
        manifest = RecoveryManifest()
        node_id = uuid4()

        op = manifest.log_collapse(
            node_id=node_id,
            original_node_ids=[uuid4(), uuid4()],
            collapsed_node_id=uuid4(),
            original_tokens=1000,
            chain_description="Test chain",
        )

        assert len(manifest.operations) == 1
        assert op.chain_description == "Test chain"

    def test_log_summarize(self):
        """Test logging summarize operation."""
        manifest = RecoveryManifest()

        op = manifest.log_summarize(
            node_id=uuid4(),
            original_node_ids=[uuid4()],
            summary_node_id=uuid4(),
            original_tokens=2000,
            summary_tokens=200,
            method="task_aware",
        )

        assert len(manifest.operations) == 1
        assert op.method == "task_aware"

    def test_log_evict(self):
        """Test logging evict operation."""
        manifest = RecoveryManifest()

        op = manifest.log_evict(
            node_id=uuid4(),
            original_tokens=500,
            reason="expired",
        )

        assert len(manifest.operations) == 1
        assert op.reason == "expired"

    def test_get_operations_for_node(self):
        """Test getting operations for a specific node."""
        manifest = RecoveryManifest()
        node_id = uuid4()

        # Log multiple operations for same node
        manifest.log_externalize(
            node_id=node_id,
            external_uri="memory://1",
            original_tokens=100,
            content_hash="h1",
        )
        manifest.log_externalize(
            node_id=uuid4(),  # Different node
            external_uri="memory://2",
            original_tokens=100,
            content_hash="h2",
        )

        ops = manifest.get_operations_for_node(node_id)
        assert len(ops) == 1
        assert ops[0].node_id == node_id

    def test_can_recover_node(self):
        """Test checking if node can be recovered."""
        manifest = RecoveryManifest()
        recoverable_node = uuid4()
        unrecoverable_node = uuid4()

        # Log recoverable operation
        manifest.log_externalize(
            node_id=recoverable_node,
            external_uri="memory://test",
            original_tokens=100,
            content_hash="hash",
        )

        # Log unrecoverable operation
        manifest.log_summarize(
            node_id=unrecoverable_node,
            original_node_ids=[uuid4()],
            summary_node_id=uuid4(),
            original_tokens=1000,
            summary_tokens=100,
        )

        assert manifest.can_recover_node(recoverable_node) is True
        assert manifest.can_recover_node(unrecoverable_node) is False

    def test_can_recover_node_no_operations(self):
        """Test recovery check for node with no operations."""
        manifest = RecoveryManifest()
        assert manifest.can_recover_node(uuid4()) is True

    def test_get_recoverable_nodes(self):
        """Test getting all recoverable nodes."""
        manifest = RecoveryManifest()
        node1 = uuid4()
        node2 = uuid4()

        manifest.log_externalize(
            node_id=node1,
            external_uri="memory://1",
            original_tokens=100,
            content_hash="h1",
        )
        manifest.log_summarize(
            node_id=node2,
            original_node_ids=[uuid4()],
            summary_node_id=uuid4(),
            original_tokens=500,
            summary_tokens=50,
        )

        recoverable = manifest.get_recoverable_nodes()
        assert node1 in recoverable
        assert node2 not in recoverable

    def test_get_stats(self):
        """Test getting manifest statistics."""
        manifest = RecoveryManifest()

        manifest.log_externalize(
            node_id=uuid4(),
            external_uri="memory://1",
            original_tokens=1000,
            content_hash="h1",
        )
        manifest.log_externalize(
            node_id=uuid4(),
            external_uri="memory://2",
            original_tokens=500,
            content_hash="h2",
        )
        manifest.log_summarize(
            node_id=uuid4(),
            original_node_ids=[uuid4()],
            summary_node_id=uuid4(),
            original_tokens=2000,
            summary_tokens=200,
        )

        stats = manifest.get_stats()
        assert stats.total_operations == 3
        assert stats.nodes_affected == 3
        assert stats.recoverable_operations == 2
        assert stats.operations_by_type["externalize"] == 2
        assert stats.operations_by_type["summarize"] == 1

    def test_clear(self):
        """Test clearing the manifest."""
        manifest = RecoveryManifest()
        manifest.log_externalize(
            node_id=uuid4(),
            external_uri="memory://1",
            original_tokens=100,
            content_hash="h1",
        )

        manifest.clear()

        assert len(manifest.operations) == 0
        stats = manifest.get_stats()
        assert stats.total_operations == 0

    def test_serialization(self):
        """Test manifest serialization and deserialization."""
        manifest = RecoveryManifest(session_id="test")
        node_id = uuid4()

        manifest.log_externalize(
            node_id=node_id,
            external_uri="memory://test",
            original_tokens=500,
            content_hash="hash123",
        )

        # Serialize
        data = manifest.to_dict()
        assert data["session_id"] == "test"
        assert len(data["operations"]) == 1

        # Deserialize
        restored = RecoveryManifest.from_dict(data)
        assert restored.session_id == "test"
        assert len(restored.operations) == 1
        assert restored.operations[0].node_id == node_id
