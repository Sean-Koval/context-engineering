"""Tests for TaskRelevanceCompression strategy."""

from __future__ import annotations

import pytest

from context_compression.recovery import RecoveryManifest
from context_compression.recovery.operations import TaskRelevanceOperation
from context_compression.strategies.compaction import TaskRelevanceCompression
from context_compression.types import CompressionTier
from context_core.graph import ContextGraph
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, NodeType
from context_core.semantic import SemanticIndex
from context_core.semantic.embeddings import MockEmbeddingModel


class TestTaskRelevanceProperties:
    """Tests for strategy properties."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with default settings."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(semantic_index=index)

    def test_strategy_name(self, strategy: TaskRelevanceCompression) -> None:
        """Test strategy name property."""
        assert strategy.name == "task_relevance_compression"

    def test_strategy_tier(self, strategy: TaskRelevanceCompression) -> None:
        """Test strategy tier property."""
        assert strategy.tier == CompressionTier.COMPACTION

    def test_strategy_priority(self, strategy: TaskRelevanceCompression) -> None:
        """Test strategy priority is after schema compression."""
        assert strategy.priority == 30


class TestExtractTaskContext:
    """Tests for task context extraction."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with default settings."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(semantic_index=index)

    def test_extracts_system_prompt(self, strategy: TaskRelevanceCompression) -> None:
        """Test extraction includes system prompt content."""
        graph = ContextGraph()
        # Add system message
        system_node = ContextNode(
            type=NodeType.SYSTEM,
            content=Content(text="You are a Python code reviewer"),
            metadata=NodeMetadata(),
        )
        graph.add_node(system_node)

        # Add some user messages
        graph.add_message("user", "Review this code")

        task_context = strategy._extract_task_context(graph)

        assert "Python code reviewer" in task_context

    def test_extracts_recent_user_messages(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test extraction includes recent user messages."""
        graph = ContextGraph()

        # Add multiple user messages
        graph.add_message("user", "First message about testing")
        graph.add_message("assistant", "Response 1")
        graph.add_message("user", "Second message about deployment")
        graph.add_message("assistant", "Response 2")
        graph.add_message("user", "Third message about security")

        task_context = strategy._extract_task_context(graph)

        # Should include recent user messages
        assert "security" in task_context

    def test_extracts_high_importance_messages(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test extraction includes high importance messages."""
        graph = ContextGraph()

        # Add a high importance message
        graph.add_message("user", "Critical task: fix authentication", importance=1.0)
        graph.add_message("user", "Low priority chat", importance=0.1)

        task_context = strategy._extract_task_context(graph)

        assert "authentication" in task_context

    def test_empty_graph_returns_empty_context(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test empty graph returns empty task context."""
        graph = ContextGraph()
        task_context = strategy._extract_task_context(graph)
        assert task_context == ""


class TestComputeRelevanceScore:
    """Tests for relevance score computation."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with specific weights for testing."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            semantic_weight=0.5,
            recency_weight=0.3,
            importance_weight=0.2,
        )

    def test_score_range(self, strategy: TaskRelevanceCompression) -> None:
        """Test that scores are in valid range 0-1."""
        graph = ContextGraph()
        graph.add_message("user", "Test the API endpoints")

        nodes = list(graph)
        task_embedding = strategy._get_task_embedding("API testing")

        for node in nodes:
            score = strategy._compute_relevance_score(node, task_embedding, 10)
            assert 0.0 <= score <= 1.0

    def test_recency_affects_score(self, strategy: TaskRelevanceCompression) -> None:
        """Test that more recent nodes have higher recency component."""
        graph = ContextGraph()

        # Add nodes in sequence
        for i in range(10):
            graph.add_message("user", f"Message {i}")

        nodes = list(graph)
        first_node = nodes[0]
        last_node = nodes[-1]

        max_seq = max((n.sequence_number or 0) for n in nodes)

        score_first = strategy._compute_relevance_score(first_node, None, max_seq)
        score_last = strategy._compute_relevance_score(last_node, None, max_seq)

        # Last node should have higher recency contribution
        # Since semantic is neutral (None), recency should make a difference
        assert score_last > score_first

    def test_importance_affects_score(self, strategy: TaskRelevanceCompression) -> None:
        """Test that node importance affects relevance score."""
        graph = ContextGraph()

        # Add nodes with different importance
        graph.add_message("user", "Low importance", importance=0.1)
        graph.add_message("user", "High importance", importance=1.0)

        nodes = list(graph)
        low_node = nodes[0]
        high_node = nodes[1]

        max_seq = max((n.sequence_number or 0) for n in nodes)

        score_low = strategy._compute_relevance_score(low_node, None, max_seq)
        score_high = strategy._compute_relevance_score(high_node, None, max_seq)

        # High importance should contribute more
        assert score_high > score_low


class TestCompressOffTaskContent:
    """Tests for basic compression functionality."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with settings that make compression likely."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.8,  # High threshold to trigger compression
            min_age_to_compress=2,  # Only protect 2 recent nodes
        )

    @pytest.fixture
    def graph_with_off_task_content(self) -> ContextGraph:
        """Create graph with content that should be considered off-task."""
        graph = ContextGraph()

        # Add system prompt about Python
        system_node = ContextNode(
            type=NodeType.SYSTEM,
            content=Content(text="You are a Python code reviewer"),
            metadata=NodeMetadata(),
        )
        graph.add_node(system_node)

        # Add on-task messages
        graph.add_message("user", "Review this Python function")

        # Add off-task messages (about cooking)
        graph.add_message("user", "What's a good recipe for lasagna?")
        graph.add_message(
            "assistant", "Here's how to make lasagna with pasta sheets..."
        )

        # Add more on-task content (recent, should be protected)
        graph.add_message("user", "Check the error handling in Python")
        graph.add_message("assistant", "The Python code looks good")

        # Set token counts
        for node in graph:
            node.token_count = 50

        return graph

    def test_compresses_off_task_nodes(
        self,
        strategy: TaskRelevanceCompression,
        graph_with_off_task_content: ContextGraph,
    ) -> None:
        """Test that off-task content is compressed."""
        manifest = RecoveryManifest()

        result = strategy.compress(graph_with_off_task_content, manifest)

        assert result.success is True
        assert result.strategy_name == "task_relevance_compression"
        assert result.nodes_compressed > 0

    def test_compression_updates_node_level(
        self,
        strategy: TaskRelevanceCompression,
        graph_with_off_task_content: ContextGraph,
    ) -> None:
        """Test that compressed nodes have updated compression level."""
        manifest = RecoveryManifest()

        strategy.compress(graph_with_off_task_content, manifest)

        # Find compressed nodes
        compressed_count = sum(
            1
            for node in graph_with_off_task_content
            if node.compression_level == CompressionLevel.COMPACTED
        )

        assert compressed_count > 0

    def test_compression_logs_operations(
        self,
        strategy: TaskRelevanceCompression,
        graph_with_off_task_content: ContextGraph,
    ) -> None:
        """Test that compression logs operations to manifest."""
        manifest = RecoveryManifest()

        strategy.compress(graph_with_off_task_content, manifest)

        # Should have logged at least one operation
        assert len(manifest.operations) > 0

        # Check operation type
        for op in manifest.operations:
            assert op.op_type == "task_relevance"
            assert hasattr(op, "relevance_score")
            assert hasattr(op, "original_content")


class TestPreservesRecentNodes:
    """Tests for protecting recent nodes from compression."""

    def test_preserves_recent_n_nodes(self) -> None:
        """Test that min_age_to_compress recent nodes are preserved."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,  # Very high to force compression
            min_age_to_compress=5,  # Protect 5 most recent
        )

        graph = ContextGraph()
        # Add 10 messages
        for i in range(10):
            graph.add_message("user", f"Message number {i}")
            graph.add_message("assistant", f"Response {i}")

        for node in graph:
            node.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Get nodes by sequence
        nodes = sorted(graph, key=lambda n: n.sequence_number or 0)
        recent_5 = nodes[-5:]

        # Recent 5 should not be compressed
        for node in recent_5:
            assert node.compression_level == CompressionLevel.FULL


class TestPreservesHighRelevance:
    """Tests for preserving high-relevance content."""

    def test_preserves_nodes_above_threshold(self) -> None:
        """Test that nodes above relevance threshold are not compressed."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.0,  # Very low - nothing should be compressed
            min_age_to_compress=0,  # No protection by recency
        )

        graph = ContextGraph()
        graph.add_message("user", "Test message 1")
        graph.add_message("user", "Test message 2")

        for node in graph:
            node.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # With threshold=0, nothing should be below threshold
        assert result.nodes_compressed == 0


class TestRespectsTargetTokens:
    """Tests for target token limit."""

    def test_stops_at_target_tokens(self) -> None:
        """Test compression stops when target tokens reached."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,  # High to force many compressions
            min_age_to_compress=1,
        )

        graph = ContextGraph()
        # Add many messages
        for i in range(20):
            msg = graph.add_message(
                "user", f"Some content for message {i} with extra words"
            )
            msg.token_count = 100

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest, target_tokens=100)

        # Should have stopped early (saved around 100 tokens, not all possible)
        assert result.tokens_saved <= 200  # Some buffer for estimation
        assert result.nodes_compressed < 19  # Not all nodes


class TestLogsOperations:
    """Tests for recovery manifest population."""

    def test_logs_task_relevance_operations(self) -> None:
        """Test that task relevance operations are logged correctly."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=1,
        )

        graph = ContextGraph()
        graph.add_message("user", "Old off-task content about cooking recipes")
        graph.add_message("user", "Recent on-task message")

        for node in graph:
            node.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Check operations
        for op in manifest.operations:
            assert op.op_type == "task_relevance"
            assert isinstance(op, TaskRelevanceOperation)
            assert 0.0 <= op.relevance_score <= 1.0
            assert isinstance(op.original_content, str)
            assert isinstance(op.compressed_content, str)
            assert op.original_tokens >= 0
            assert op.compressed_tokens >= 0

    def test_operation_stores_original_content(self) -> None:
        """Test that operations store original content for recovery."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=0,
        )

        graph = ContextGraph()
        original_text = "This is the original message content that should be stored"
        graph.add_message("user", original_text)
        list(graph)[0].token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        if manifest.operations:
            op = manifest.operations[0]
            assert op.original_content == original_text
            assert op.is_recoverable is True


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with default settings."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(semantic_index=index)

    def test_empty_graph(self, strategy: TaskRelevanceCompression) -> None:
        """Test handling of empty graph."""
        graph = ContextGraph()
        manifest = RecoveryManifest()

        result = strategy.compress(graph, manifest)

        assert result.success is True
        assert result.nodes_compressed == 0
        assert result.tokens_saved == 0

    def test_no_task_context(self, strategy: TaskRelevanceCompression) -> None:
        """Test handling when no clear task context exists."""
        graph = ContextGraph()

        # Add only tool results with no clear task
        call = graph.add_tool_call("some_tool", {"arg": "value"})
        graph.add_tool_result(call.id, {"result": "data"})

        for node in graph:
            node.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should handle gracefully
        assert result.success is True

    def test_all_relevant_content(self) -> None:
        """Test when all content is relevant (no compression needed)."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.0,  # Nothing below this
            min_age_to_compress=0,
        )

        graph = ContextGraph()
        graph.add_message("user", "Relevant message 1")
        graph.add_message("user", "Relevant message 2")

        for node in graph:
            node.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # No compression should occur
        assert result.nodes_compressed == 0
        assert result.tokens_saved == 0

    def test_pinned_nodes_not_compressed(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that pinned nodes are never compressed."""
        graph = ContextGraph()

        # Add a pinned node
        msg = graph.add_message("user", "This is pinned and should not be compressed")
        msg.metadata.pinned = True
        msg.token_count = 100

        # Add more nodes to get past min_age_to_compress
        for i in range(10):
            graph.add_message("user", f"Regular message {i}")

        for node in graph:
            if node.token_count is None:
                node.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Find the pinned node
        pinned_node = next(n for n in graph if n.metadata.pinned)
        assert pinned_node.compression_level == CompressionLevel.FULL

    def test_already_compressed_nodes_skipped(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that already compressed nodes are skipped."""
        graph = ContextGraph()

        # Add a pre-compressed node
        msg = graph.add_message("user", "Already compressed")
        msg.compression_level = CompressionLevel.COMPACTED
        msg.token_count = 50

        # Add more nodes
        for i in range(10):
            graph.add_message("user", f"Message {i}")

        for node in graph:
            if node.token_count is None:
                node.token_count = 50

        manifest = RecoveryManifest()
        initial_level = msg.compression_level

        strategy.compress(graph, manifest)

        # Pre-compressed node should still be COMPACTED (not changed)
        assert msg.compression_level == initial_level

    def test_system_nodes_not_compressed(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that system nodes are never compressed."""
        graph = ContextGraph()

        # Add system node
        system_node = ContextNode(
            type=NodeType.SYSTEM,
            content=Content(text="Important system instructions"),
            metadata=NodeMetadata(),
        )
        graph.add_node(system_node)
        system_node.token_count = 100

        # Add many messages to get past min_age_to_compress
        for i in range(10):
            msg = graph.add_message("user", f"Message {i}")
            msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # System node should not be compressed
        assert system_node.compression_level == CompressionLevel.FULL


class TestCanApply:
    """Tests for can_apply method."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with default settings."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.5,
            min_age_to_compress=2,
        )

    def test_can_apply_empty_graph(self, strategy: TaskRelevanceCompression) -> None:
        """Test can_apply returns False for empty graph."""
        graph = ContextGraph()
        assert strategy.can_apply(graph) is False

    def test_can_apply_too_few_nodes(self, strategy: TaskRelevanceCompression) -> None:
        """Test can_apply returns False when fewer nodes than min_age_to_compress."""
        graph = ContextGraph()
        graph.add_message("user", "Only one message")

        assert strategy.can_apply(graph) is False

    def test_can_apply_with_eligible_nodes(self) -> None:
        """Test can_apply returns True when there are nodes to compress."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        strategy = TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,  # High threshold ensures nodes are below
            min_age_to_compress=1,
        )

        graph = ContextGraph()
        # Add enough nodes
        for i in range(5):
            graph.add_message("user", f"Message {i}")

        assert strategy.can_apply(graph) is True


class TestEstimateSavings:
    """Tests for savings estimation."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with settings to trigger compression."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=1,
        )

    def test_estimate_savings_returns_non_negative(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test estimate_savings returns non-negative value."""
        graph = ContextGraph()
        for i in range(5):
            msg = graph.add_message("user", f"Some message content {i}")
            msg.token_count = 50

        savings = strategy.estimate_savings(graph)
        assert savings >= 0

    def test_estimate_savings_empty_graph(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test estimate_savings returns 0 for empty graph."""
        graph = ContextGraph()
        savings = strategy.estimate_savings(graph)
        assert savings == 0

    def test_estimate_savings_respects_targets(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test estimate_savings respects target_node_ids."""
        graph = ContextGraph()
        nodes = []
        for i in range(5):
            msg = graph.add_message("user", f"Message content {i}")
            msg.token_count = 50
            nodes.append(msg)

        # Estimate for all nodes
        savings_all = strategy.estimate_savings(graph)

        # Estimate for subset
        savings_subset = strategy.estimate_savings(graph, target_node_ids=[nodes[0].id])

        # Subset should be <= all
        assert savings_subset <= savings_all


class TestToolCallAndResultCompression:
    """Tests for compression of tool calls and results."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with high threshold."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=1,
        )

    def test_compresses_tool_call_args(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that tool call arguments are compressed correctly."""
        graph = ContextGraph()

        # Add off-task tool call
        call = graph.add_tool_call(
            "get_weather", {"city": "London", "date": "2024-01-01"}
        )
        call.token_count = 50

        # Add recent node to protect
        graph.add_message("user", "Recent message")

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Check if tool call was compressed
        if call.compression_level == CompressionLevel.COMPACTED:
            # Args should be replaced with compressed marker
            assert call.content.tool_args is not None
            assert "_compressed" in call.content.tool_args

    def test_compresses_tool_result(self, strategy: TaskRelevanceCompression) -> None:
        """Test that tool results are compressed correctly."""
        graph = ContextGraph()

        # Add off-task tool result
        call = graph.add_tool_call("search", {"query": "recipes"})
        result = graph.add_tool_result(
            call.id,
            {"results": [{"title": "Recipe 1"}, {"title": "Recipe 2"}]},
        )
        call.token_count = 30
        result.token_count = 100

        # Add recent node to protect
        graph.add_message("user", "Recent message")

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        # Check if result was compressed
        if result.compression_level == CompressionLevel.COMPACTED:
            # Output should be a compressed string
            assert isinstance(result.content.tool_output, str)
            assert "[Off-task" in result.content.tool_output


class TestCompressedContentFormat:
    """Tests for the format of compressed content."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with high threshold."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=0,
        )

    def test_message_compressed_format(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that compressed messages have correct format."""
        graph = ContextGraph()
        msg = graph.add_message("user", "This is a test message about cooking")
        msg.token_count = 50

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        if msg.compression_level == CompressionLevel.COMPACTED:
            assert msg.content.text is not None
            assert msg.content.text.startswith("[Off-task")
            assert "user" in msg.content.text or "message" in msg.content.text

    def test_long_content_truncated(self, strategy: TaskRelevanceCompression) -> None:
        """Test that long content is truncated in compressed form."""
        graph = ContextGraph()
        long_content = "x" * 500
        msg = graph.add_message("user", long_content)
        msg.token_count = 150

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        if msg.compression_level == CompressionLevel.COMPACTED:
            # Should be much shorter than original
            assert msg.content.text is not None
            assert len(msg.content.text) < len(long_content)
            assert "..." in msg.content.text


class TestTokenTracking:
    """Tests for token count tracking."""

    @pytest.fixture
    def strategy(self) -> TaskRelevanceCompression:
        """Create strategy with high threshold."""
        model = MockEmbeddingModel(dimension=64)
        index = SemanticIndex(model)
        return TaskRelevanceCompression(
            semantic_index=index,
            relevance_threshold=0.99,
            min_age_to_compress=0,
        )

    def test_updates_token_counts(self, strategy: TaskRelevanceCompression) -> None:
        """Test that token counts are updated on compressed nodes."""
        graph = ContextGraph()
        msg = graph.add_message("user", "Some content that will be compressed")
        msg.token_count = 100

        manifest = RecoveryManifest()
        strategy.compress(graph, manifest)

        if msg.compression_level == CompressionLevel.COMPACTED:
            # Original tokens stored
            assert msg.content.original_tokens == 100
            # Compressed tokens set
            assert msg.content.compressed_tokens is not None
            assert msg.content.compressed_tokens < 100
            # Node token count updated
            assert msg.token_count == msg.content.compressed_tokens

    def test_result_reports_correct_savings(
        self, strategy: TaskRelevanceCompression
    ) -> None:
        """Test that compression result reports correct savings."""
        graph = ContextGraph()

        # Add messages with known token counts
        for i in range(5):
            msg = graph.add_message("user", f"Message {i} with some content")
            msg.token_count = 50

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Tokens saved should equal original - compressed
        assert result.tokens_saved == result.original_tokens - result.compressed_tokens
