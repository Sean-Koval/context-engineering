"""Tests for HierarchicalSummarization strategy."""

from __future__ import annotations

import pytest

from context_compression.recovery import RecoveryManifest
from context_compression.recovery.operations import SummarizeOperation
from context_compression.strategies.summarization import (
    HierarchicalSummarization,
    LLMSummarizer,
    MockSummarizer,
)
from context_compression.types import CompressionTier
from context_core.graph import ContextGraph
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, NodeType


class TestHierarchicalProperties:
    """Tests for strategy properties."""

    @pytest.fixture
    def strategy(self) -> HierarchicalSummarization:
        """Create strategy with default settings."""
        return HierarchicalSummarization(summarizer=MockSummarizer())

    def test_strategy_name(self, strategy: HierarchicalSummarization) -> None:
        """Test strategy name property."""
        assert strategy.name == "hierarchical_summarization"

    def test_strategy_tier(self, strategy: HierarchicalSummarization) -> None:
        """Test strategy tier property is SUMMARIZATION."""
        assert strategy.tier == CompressionTier.SUMMARIZATION

    def test_strategy_priority(self, strategy: HierarchicalSummarization) -> None:
        """Test strategy priority."""
        assert strategy.priority == 10


class TestMockSummarizer:
    """Tests for MockSummarizer."""

    def test_mock_summarizer_creates_deterministic_output(self) -> None:
        """Test that MockSummarizer produces predictable summaries."""
        summarizer = MockSummarizer()

        texts = ["Hello world", "How are you?"]
        result = summarizer.summarize(texts, max_tokens=100)

        assert "[Summary of 2 messages" in result
        assert "23 chars]" in result  # len("Hello world") + len("How are you?") = 23

    def test_mock_summarizer_reflects_input_count(self) -> None:
        """Test that MockSummarizer reflects the number of inputs."""
        summarizer = MockSummarizer()

        texts = ["A", "B", "C", "D", "E"]
        result = summarizer.summarize(texts, max_tokens=50)

        assert "[Summary of 5 messages" in result

    def test_mock_summarizer_ignores_instruction(self) -> None:
        """Test that MockSummarizer ignores the instruction parameter."""
        summarizer = MockSummarizer()

        result1 = summarizer.summarize(["test"], 100, instruction=None)
        result2 = summarizer.summarize(["test"], 100, instruction="Custom instruction")

        assert result1 == result2


class TestGroupsIntoChunks:
    """Tests for chunking logic."""

    @pytest.fixture
    def strategy(self) -> HierarchicalSummarization:
        """Create strategy with specific chunk settings."""
        return HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,  # 3 messages per chunk
            chunk_threshold=500,  # Or 500 tokens
            preserve_recent=0,  # Don't protect any for this test
        )

    def test_groups_by_chunk_size(self, strategy: HierarchicalSummarization) -> None:
        """Test that nodes are grouped by chunk_size."""
        graph = ContextGraph()

        # Add 7 messages with small token counts
        for i in range(7):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 10

        eligible = strategy._get_eligible_nodes(graph, None)
        chunks = strategy._group_into_chunks(eligible)

        # Should have 3 chunks: [0,1,2], [3,4,5], [6]
        assert len(chunks) == 3
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 1

    def test_groups_by_token_threshold(self) -> None:
        """Test that nodes are grouped when token threshold is reached."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=100,  # Large size, so threshold triggers first
            chunk_threshold=200,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add 5 messages with 100 tokens each
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100

        eligible = strategy._get_eligible_nodes(graph, None)
        chunks = strategy._group_into_chunks(eligible)

        # Should create chunks based on 200 token threshold
        # Each message is 100 tokens, so 2 per chunk
        # 5 messages -> chunks of 2, 2, 1
        assert len(chunks) == 3

    def test_empty_nodes_returns_empty_chunks(
        self, strategy: HierarchicalSummarization
    ) -> None:
        """Test that empty node list returns empty chunks."""
        chunks = strategy._group_into_chunks([])
        assert chunks == []


class TestCreatesSummaryNodes:
    """Tests for summary node creation."""

    @pytest.fixture
    def strategy(self) -> HierarchicalSummarization:
        """Create strategy with small chunk size."""
        return HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

    def test_creates_summary_nodes(self, strategy: HierarchicalSummarization) -> None:
        """Test that summary nodes are created."""
        graph = ContextGraph()

        # Add 5 messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_created > 0

        # Check that SUMMARY nodes exist
        summary_nodes = [n for n in graph if n.type == NodeType.SUMMARY]
        assert len(summary_nodes) > 0

    def test_summary_node_has_correct_properties(
        self, strategy: HierarchicalSummarization
    ) -> None:
        """Test that summary nodes have correct properties."""
        graph = ContextGraph()

        # Add messages with specific properties
        for i in range(3):
            msg = graph.add_message("user", f"Message {i}", importance=0.5 + i * 0.1)
            msg.metadata.tags.add(f"tag_{i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        summary_nodes = [n for n in graph if n.type == NodeType.SUMMARY]
        assert len(summary_nodes) == 1

        summary = summary_nodes[0]
        assert summary.type == NodeType.SUMMARY
        assert summary.compression_level == CompressionLevel.SUMMARIZED
        assert summary.content.summary_method == "hierarchical"
        assert "hierarchical_summary" in summary.metadata.tags


class TestPreservesRecentMessages:
    """Tests for preserve_recent functionality."""

    def test_preserves_recent_n_messages(self) -> None:
        """Test that preserve_recent N messages are not summarized."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=3,  # Protect 3 most recent
        )

        graph = ContextGraph()

        # Add 10 messages
        for i in range(10):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Get remaining message nodes
        message_nodes = [n for n in graph if n.type == NodeType.MESSAGE]

        # Should have exactly 3 message nodes remaining (the protected ones)
        assert len(message_nodes) == 3

        # They should be the last 3 added (highest sequence numbers)
        sequences = sorted([n.sequence_number for n in message_nodes])
        assert sequences == [7, 8, 9]

    def test_all_protected_when_fewer_than_preserve_recent(self) -> None:
        """Test that all nodes are protected if fewer than preserve_recent."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=10,  # Protect 10
        )

        graph = ContextGraph()

        # Add only 5 messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should not compress anything (all protected)
        assert result.nodes_removed == 0


class TestMessageLevelSummarization:
    """Tests for message-level summarization of long messages."""

    def test_long_messages_included_in_chunks(self) -> None:
        """Test that long messages are included in chunk-level summarization."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            message_threshold=100,  # Messages > 100 tokens are "long"
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add a mix of short and long messages
        short_msg = graph.add_message("user", "Short message")
        short_msg.token_count = 50

        long_msg = graph.add_message("user", "Long message " * 50)
        long_msg.token_count = 200

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Both should be processed in chunks
        assert result.nodes_processed == 2
        assert result.nodes_removed == 2


class TestChunkLevelSummarization:
    """Tests for chunk-level summarization."""

    def test_chunks_summarized_together(self) -> None:
        """Test that chunks of messages are summarized together."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add 6 messages (2 chunks of 3)
        for i in range(6):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should create 2 summary nodes
        assert result.nodes_created == 2

        # Should remove 6 original nodes
        assert result.nodes_removed == 6


class TestCreatesSummarizesEdges:
    """Tests for SUMMARIZES edge creation."""

    def test_creates_summarizes_edges(self) -> None:
        """Test that SUMMARIZES edges are created between summary and originals."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add 3 messages
        node_ids: list = []
        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50
            node_ids.append(msg.id)

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Get the summary node
        summary_nodes = [n for n in graph if n.type == NodeType.SUMMARY]
        assert len(summary_nodes) == 1

        summary = summary_nodes[0]

        # Check summarized_node_ids in content
        assert summary.content.summarized_node_ids is not None
        assert len(summary.content.summarized_node_ids) == 3


class TestRemovesOriginalNodes:
    """Tests for removing original nodes after summarization."""

    def test_removes_original_nodes(self) -> None:
        """Test that original nodes are removed after summarization."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add 3 messages and record their IDs
        original_ids: list = []
        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50
            original_ids.append(msg.id)

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Original nodes should be removed
        for node_id in original_ids:
            assert graph.get_node(node_id) is None


class TestLogsOperations:
    """Tests for recovery manifest population."""

    def test_logs_summarize_operations(self) -> None:
        """Test that SummarizeOperation is logged to manifest."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Should have logged at least one operation
        assert len(manifest.operations) > 0

        # Check operation type
        for op in manifest.operations:
            assert op.op_type == "summarize"
            assert isinstance(op, SummarizeOperation)
            assert op.method == "hierarchical"
            assert len(op.original_node_ids) > 0
            assert op.original_tokens > 0
            assert op.summary_tokens >= 0

    def test_operation_stores_summary_text(self) -> None:
        """Test that operations store the summary text."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        op = manifest.operations[0]
        assert op.summary_text is not None
        assert len(op.summary_text) > 0
        assert "[Summary of" in op.summary_text  # From MockSummarizer


class TestRespectsTargetTokens:
    """Tests for target token limit."""

    def test_stops_at_target_tokens(self) -> None:
        """Test compression stops when target tokens reached."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add 10 messages with 100 tokens each
        for i in range(10):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100

        manifest = RecoveryManifest()
        # Target only 100 tokens saved
        result = strategy.compress(graph, manifest, target_tokens=100)

        # Should have stopped early
        assert result.nodes_removed < 10

    def test_continues_without_target(self) -> None:
        """Test compression continues when no target specified."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(6):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest, target_tokens=None)

        # Should process all nodes
        assert result.nodes_removed == 6


class TestEmptyGraph:
    """Tests for empty graph edge case."""

    def test_empty_graph_returns_success(self) -> None:
        """Test handling of empty graph."""
        strategy = HierarchicalSummarization(summarizer=MockSummarizer())

        graph = ContextGraph()
        manifest = RecoveryManifest()

        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_processed == 0
        assert result.tokens_saved == 0


class TestTooFewMessages:
    """Tests for too few messages edge case."""

    def test_can_apply_returns_false(self) -> None:
        """Test can_apply returns False when fewer than chunk_size messages."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=10,  # Need 10 messages
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add only 5 messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        assert strategy.can_apply(graph) is False

    def test_can_apply_returns_true_with_enough_messages(self) -> None:
        """Test can_apply returns True with enough messages."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        assert strategy.can_apply(graph) is True


class TestPinnedNodesPreserved:
    """Tests for pinned node preservation."""

    def test_pinned_nodes_not_summarized(self) -> None:
        """Test that pinned nodes are never summarized."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add a pinned node
        pinned = graph.add_message("user", "This is pinned and must stay")
        pinned.metadata.pinned = True
        pinned.token_count = 100
        pinned_id = pinned.id

        # Add more nodes
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Pinned node should still exist
        assert graph.get_node(pinned_id) is not None


class TestAlreadySummarizedSkipped:
    """Tests for skipping already summarized nodes."""

    def test_already_summarized_nodes_skipped(self) -> None:
        """Test that already summarized nodes are skipped."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add a pre-summarized node
        summarized = graph.add_message("user", "Already summarized")
        summarized.compression_level = CompressionLevel.SUMMARIZED
        summarized.token_count = 50
        summarized_id = summarized.id

        # Add more nodes
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Pre-summarized node should still exist
        assert graph.get_node(summarized_id) is not None

    def test_summary_nodes_not_re_summarized(self) -> None:
        """Test that SUMMARY type nodes are not re-summarized."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add an existing summary node
        summary_node = ContextNode(
            type=NodeType.SUMMARY,
            content=Content(
                text="Existing summary",
                summary_method="hierarchical",
            ),
            metadata=NodeMetadata(),
        )
        graph.add_node(summary_node)
        summary_node.token_count = 30
        summary_id = summary_node.id

        # Add regular messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Existing summary should still exist (not re-summarized)
        assert graph.get_node(summary_id) is not None


class TestSystemNodesPreserved:
    """Tests for system node preservation."""

    def test_system_nodes_not_summarized(self) -> None:
        """Test that system nodes are never summarized."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add a system node
        system_node = ContextNode(
            type=NodeType.SYSTEM,
            content=Content(text="Important system instructions"),
            metadata=NodeMetadata(),
        )
        graph.add_node(system_node)
        system_node.token_count = 100
        system_id = system_node.id

        # Add regular messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # System node should still exist
        assert graph.get_node(system_id) is not None


class TestEstimateSavings:
    """Tests for savings estimation."""

    @pytest.fixture
    def strategy(self) -> HierarchicalSummarization:
        """Create strategy."""
        return HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

    def test_estimate_savings_returns_non_negative(
        self, strategy: HierarchicalSummarization
    ) -> None:
        """Test estimate_savings returns non-negative value."""
        graph = ContextGraph()

        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100

        savings = strategy.estimate_savings(graph)
        assert savings >= 0

    def test_estimate_savings_empty_graph(
        self, strategy: HierarchicalSummarization
    ) -> None:
        """Test estimate_savings returns 0 for empty graph."""
        graph = ContextGraph()
        savings = strategy.estimate_savings(graph)
        assert savings == 0

    def test_estimate_savings_respects_targets(
        self, strategy: HierarchicalSummarization
    ) -> None:
        """Test estimate_savings respects target_node_ids."""
        graph = ContextGraph()

        nodes = []
        for i in range(6):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100
            nodes.append(msg)

        # Estimate for all nodes
        savings_all = strategy.estimate_savings(graph)

        # Estimate for subset
        subset_ids = [nodes[0].id, nodes[1].id, nodes[2].id]
        savings_subset = strategy.estimate_savings(graph, target_node_ids=subset_ids)

        # Subset should be <= all
        assert savings_subset <= savings_all


class TestTokenTracking:
    """Tests for token count tracking."""

    def test_result_reports_correct_metrics(self) -> None:
        """Test that compression result reports correct metrics."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add messages with known token counts
        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 100

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Original tokens should be 300 (3 * 100)
        assert result.original_tokens == 300

        # Compressed tokens should be less
        assert result.compressed_tokens < result.original_tokens

        # Tokens saved should equal original - compressed
        assert result.tokens_saved == result.original_tokens - result.compressed_tokens


class TestToolCallsAndResults:
    """Tests for handling tool calls and results."""

    def test_tool_calls_can_be_summarized(self) -> None:
        """Test that tool calls are included in summarization."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add tool call and result
        call = graph.add_tool_call("search", {"query": "test"})
        call.token_count = 50

        result = graph.add_tool_result(call.id, {"results": []})
        result.token_count = 50

        manifest = RecoveryManifest()
        compression_result = strategy.compress(graph, manifest)

        assert compression_result.nodes_processed == 2
        assert compression_result.nodes_removed == 2

    def test_tool_text_extraction(self) -> None:
        """Test that tool call/result text is correctly extracted."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        call = graph.add_tool_call("search", {"query": "test query"})
        call.token_count = 50

        text = strategy._get_node_text(call)
        assert "search" in text
        assert "test query" in text or "query" in text


class TestEvictedNodesSkipped:
    """Tests for skipping evicted nodes."""

    def test_evicted_nodes_skipped(self) -> None:
        """Test that evicted nodes are skipped."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add an evicted node
        evicted = graph.add_message("user", "Evicted content")
        evicted.compression_level = CompressionLevel.EVICTED
        evicted.token_count = 50
        evicted_id = evicted.id

        # Add regular messages
        for i in range(5):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Evicted node should still exist (not processed)
        assert graph.get_node(evicted_id) is not None


class TestLLMSummarizerProtocol:
    """Tests for LLMSummarizer protocol."""

    def test_mock_summarizer_implements_protocol(self) -> None:
        """Test that MockSummarizer implements LLMSummarizer protocol."""
        summarizer: LLMSummarizer = MockSummarizer()
        result = summarizer.summarize(["test"], 100)
        assert isinstance(result, str)

    def test_custom_summarizer_can_be_used(self) -> None:
        """Test that a custom summarizer can be used."""

        class CustomSummarizer:
            def summarize(
                self,
                texts: list[str],
                max_tokens: int,
                instruction: str | None = None,
            ) -> str:
                return f"Custom summary of {len(texts)} items"

        strategy = HierarchicalSummarization(
            summarizer=CustomSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()
        for i in range(2):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        summary_nodes = [n for n in graph if n.type == NodeType.SUMMARY]
        assert len(summary_nodes) == 1
        assert "Custom summary" in summary_nodes[0].content.text


class TestCompressionIsIrreversible:
    """Tests confirming summarization is irreversible."""

    def test_is_recoverable_false(self) -> None:
        """Test that is_recoverable is False for summarization."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()
        for i in range(2):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.is_recoverable is False

    def test_operation_is_recoverable_false(self) -> None:
        """Test that SummarizeOperation.is_recoverable is False."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=2,
            preserve_recent=0,
        )

        graph = ContextGraph()
        for i in range(2):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        for op in manifest.operations:
            assert op.is_recoverable is False


class TestMetadataPreservation:
    """Tests for metadata preservation in summaries."""

    def test_preserves_max_importance(self) -> None:
        """Test that summary preserves max importance from chunk."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        # Add messages with different importance
        for i, importance in enumerate([0.3, 0.9, 0.5]):
            msg = graph.add_message("user", f"Message {i}", importance=importance)
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        summary = [n for n in graph if n.type == NodeType.SUMMARY][0]

        # Should preserve highest importance from chunk (approximately)
        # Note: compute_importance includes type weight, recency, etc.
        assert summary.metadata.importance > 0.5

    def test_preserves_tags_from_chunk(self) -> None:
        """Test that summary collects tags from all chunk nodes."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.metadata.tags.add(f"tag_{i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        summary = [n for n in graph if n.type == NodeType.SUMMARY][0]

        # Should have hierarchical_summary tag plus collected tags
        assert "hierarchical_summary" in summary.metadata.tags
        assert "tag_0" in summary.metadata.tags
        assert "tag_1" in summary.metadata.tags
        assert "tag_2" in summary.metadata.tags

    def test_preserves_entities_from_chunk(self) -> None:
        """Test that summary collects entities from all chunk nodes."""
        strategy = HierarchicalSummarization(
            summarizer=MockSummarizer(),
            chunk_size=3,
            preserve_recent=0,
        )

        graph = ContextGraph()

        for i in range(3):
            msg = graph.add_message("user", f"Message {i}")
            msg.metadata.entities.append(f"entity_{i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        summary = [n for n in graph if n.type == NodeType.SUMMARY][0]

        # Should have collected entities
        assert "entity_0" in summary.metadata.entities
        assert "entity_1" in summary.metadata.entities
        assert "entity_2" in summary.metadata.entities
