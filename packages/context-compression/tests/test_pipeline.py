"""Tests for CompressionPipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from context_compression.pipeline import CompressionPipeline
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.strategies.lossless import (
    ExternalizePayloads,
    InMemoryExternalStorage,
)
from context_compression.types import (
    CompressionResult,
    CompressionTier,
    PipelineConfig,
    PreservationRule,
)
from context_core.graph import ContextGraph
from context_core.graph.nodes import Content, ContextNode
from context_core.graph.types import NodeType


class MockStrategy(BaseCompressionStrategy):
    """Mock strategy for testing."""

    def __init__(
        self,
        name: str = "mock",
        tier: CompressionTier = CompressionTier.LOSSLESS,
        priority: int = 0,
        savings: int = 100,
    ):
        self._name_value = name
        self._tier_value = tier
        self._priority_value = priority
        self._savings = savings
        self.compress_called = False

    @property
    def _name(self) -> str:
        return self._name_value

    @property
    def _tier(self) -> CompressionTier:
        return self._tier_value

    @property
    def _priority(self) -> int:
        return self._priority_value

    def _estimate_savings_impl(self, graph, target_node_ids):
        return self._savings

    def _compress_impl(self, graph, manifest, target_node_ids, target_tokens):
        self.compress_called = True
        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=200,
            compressed_tokens=100,
            tokens_saved=self._savings,
            nodes_processed=len(list(graph)),
            nodes_compressed=1,
            is_recoverable=True,
        )

    def _can_apply_impl(self, graph):
        return True


class TestCompressionPipeline:
    """Tests for CompressionPipeline."""

    @pytest.fixture
    def simple_graph(self):
        """Create a simple graph with messages."""
        graph = ContextGraph()
        for i in range(10):
            msg = graph.add_message(role="user", content=f"Message {i}")
            msg.token_count = 50
        return graph

    @pytest.fixture
    def graph_with_tools(self):
        """Create a graph with tool calls."""
        graph = ContextGraph()
        graph.add_message(role="user", content="Read files")

        for i in range(5):
            call = graph.add_tool_call("read_file", {"path": f"/file{i}.txt"})
            call.token_count = 50
            result = graph.add_tool_result(call.id, "x" * 1000)
            result.token_count = 250

        return graph

    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = CompressionPipeline()
        assert len(pipeline.strategies) == 0
        assert len(pipeline.preservation_rules) == 4  # Default rules

    def test_create_pipeline_with_strategies(self):
        """Test creating pipeline with initial strategies."""
        strategies = [
            MockStrategy("s1", CompressionTier.LOSSLESS),
            MockStrategy("s2", CompressionTier.COMPACTION),
        ]
        pipeline = CompressionPipeline(strategies=strategies)
        assert len(pipeline.strategies) == 2

    def test_create_pipeline_custom_rules(self):
        """Test creating pipeline with custom preservation rules."""
        rules = [PreservationRule(name="custom", pinned=True)]
        pipeline = CompressionPipeline(preservation_rules=rules)
        assert len(pipeline.preservation_rules) == 1
        assert pipeline.preservation_rules[0].name == "custom"

    def test_register_strategy(self):
        """Test registering strategies."""
        pipeline = CompressionPipeline()
        strategy = MockStrategy()
        pipeline.register_strategy(strategy)

        assert len(pipeline.strategies) == 1
        assert pipeline.strategies[0] == strategy

    def test_unregister_strategy(self):
        """Test unregistering strategies."""
        pipeline = CompressionPipeline()
        pipeline.register_strategy(MockStrategy("to_remove"))
        pipeline.register_strategy(MockStrategy("keep"))

        assert pipeline.unregister_strategy("to_remove") is True
        assert len(pipeline.strategies) == 1
        assert pipeline.strategies[0].name == "keep"

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent strategy."""
        pipeline = CompressionPipeline()
        assert pipeline.unregister_strategy("nonexistent") is False

    def test_strategy_ordering(self):
        """Test strategies are ordered by tier then priority."""
        pipeline = CompressionPipeline()

        # Add out of order
        pipeline.register_strategy(
            MockStrategy("sum1", CompressionTier.SUMMARIZATION, 10)
        )
        pipeline.register_strategy(MockStrategy("loss1", CompressionTier.LOSSLESS, 20))
        pipeline.register_strategy(
            MockStrategy("comp1", CompressionTier.COMPACTION, 10)
        )
        pipeline.register_strategy(MockStrategy("loss2", CompressionTier.LOSSLESS, 10))

        names = [s.name for s in pipeline.strategies]
        assert names == ["loss2", "loss1", "comp1", "sum1"]

    def test_add_preservation_rule(self):
        """Test adding preservation rules."""
        pipeline = CompressionPipeline(preservation_rules=[])
        rule = PreservationRule(name="test", priority=50)
        pipeline.add_preservation_rule(rule)

        assert len(pipeline.preservation_rules) == 1

    def test_remove_preservation_rule(self):
        """Test removing preservation rules."""
        pipeline = CompressionPipeline(
            preservation_rules=[
                PreservationRule(name="keep"),
                PreservationRule(name="remove"),
            ]
        )

        assert pipeline.remove_preservation_rule("remove") is True
        assert len(pipeline.preservation_rules) == 1
        assert pipeline.preservation_rules[0].name == "keep"

    def test_get_preserved_nodes_recent(self, simple_graph):
        """Test preservation of recent nodes."""
        config = PipelineConfig(preserve_recent_n=5)
        pipeline = CompressionPipeline(
            preservation_rules=[],
            config=config,
        )

        preserved = pipeline.get_preserved_nodes(simple_graph)
        assert len(preserved) == 5

    def test_get_preserved_nodes_pinned(self, simple_graph):
        """Test preservation of pinned nodes."""
        # Pin first node
        for node in simple_graph:
            node.metadata.pinned = True
            break

        pipeline = CompressionPipeline(
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(simple_graph)
        assert len(preserved) >= 1

    def test_get_preserved_nodes_system(self):
        """Test preservation of system nodes."""
        graph = ContextGraph()
        graph.add_node(
            ContextNode(
                type=NodeType.SYSTEM,
                content=Content(text="System prompt"),
            )
        )
        graph.add_message(role="user", content="Hello")

        pipeline = CompressionPipeline(
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(graph)
        # System node should be preserved by default rules
        assert len(preserved) >= 1

    def test_plan_dry_run(self, simple_graph):
        """Test creating a compression plan."""
        pipeline = CompressionPipeline()
        pipeline.register_strategy(MockStrategy(savings=500))

        plan = pipeline.plan(simple_graph, target_tokens=1000)

        assert len(plan.strategies) >= 1
        assert plan.estimated_savings > 0
        assert len(plan.preservations) > 0

    def test_compress_dry_run(self, simple_graph):
        """Test dry run compression."""
        pipeline = CompressionPipeline()
        pipeline.register_strategy(MockStrategy(savings=500))

        results = pipeline.compress(simple_graph, dry_run=True)

        assert len(results) == 1
        assert results[0].strategy_name == "DRY_RUN"

    def test_compress_executes_strategies(self, simple_graph):
        """Test compression executes strategies."""
        strategy = MockStrategy()
        pipeline = CompressionPipeline(
            strategies=[strategy],
            config=PipelineConfig(preserve_recent_n=0),
        )

        results = pipeline.compress(simple_graph)

        assert strategy.compress_called is True
        assert len(results) == 1
        assert results[0].success is True

    def test_compress_respects_tier_limit(self, simple_graph):
        """Test compression respects max tier."""
        lossless = MockStrategy("lossless", CompressionTier.LOSSLESS)
        summarize = MockStrategy("summarize", CompressionTier.SUMMARIZATION)

        pipeline = CompressionPipeline(
            strategies=[lossless, summarize],
            config=PipelineConfig(preserve_recent_n=0),
        )

        pipeline.compress(
            simple_graph,
            max_tier=CompressionTier.LOSSLESS,
        )

        assert lossless.compress_called is True
        assert summarize.compress_called is False

    def test_compress_respects_target_tokens(self, simple_graph):
        """Test compression stops at target tokens."""
        s1 = MockStrategy("s1", savings=100)
        s2 = MockStrategy("s2", savings=100)
        s3 = MockStrategy("s3", savings=100)

        pipeline = CompressionPipeline(
            strategies=[s1, s2, s3],
            config=PipelineConfig(preserve_recent_n=0),
        )

        results = pipeline.compress(simple_graph, target_tokens=150)

        # Should stop after 2 strategies (200 tokens saved >= 150 target)
        total_saved = sum(r.tokens_saved for r in results)
        assert total_saved >= 150

    def test_compress_callback(self, simple_graph):
        """Test compression callback is invoked."""
        callback_results = []

        def callback(result):
            callback_results.append(result)

        pipeline = CompressionPipeline(
            strategies=[MockStrategy()],
            config=PipelineConfig(preserve_recent_n=0),
            on_compression=callback,
        )

        pipeline.compress(simple_graph)

        assert len(callback_results) == 1

    def test_compress_to_budget(self, graph_with_tools):
        """Test compressing to fit within budget."""
        storage = InMemoryExternalStorage()
        pipeline = CompressionPipeline(
            strategies=[ExternalizePayloads(storage, min_tokens=200)],
            config=PipelineConfig(preserve_recent_n=2),
        )

        # Calculate current tokens
        current = sum(n.token_count or 0 for n in graph_with_tools)

        # Compress to a lower budget
        results = pipeline.compress_to_budget(
            graph_with_tools,
            budget_tokens=current // 2,
        )

        # Should have compressed something
        total_saved = sum(r.tokens_saved for r in results)
        assert total_saved > 0

    def test_compress_to_budget_already_fits(self, simple_graph):
        """Test compress_to_budget when already within budget."""
        pipeline = CompressionPipeline()
        pipeline.register_strategy(MockStrategy())

        # Large budget that already fits
        results = pipeline.compress_to_budget(simple_graph, budget_tokens=100000)

        assert len(results) == 0


class TestPreservationRuleMatching:
    """Tests for preservation rule matching logic."""

    def test_match_by_node_type(self):
        """Test matching by node type."""
        graph = ContextGraph()
        system_node = graph.add_node(
            ContextNode(
                type=NodeType.SYSTEM,
                content=Content(text="System"),
            )
        )
        graph.add_message(role="user", content="Hello")

        pipeline = CompressionPipeline(
            preservation_rules=[
                PreservationRule(name="system", node_types=["system"]),
            ],
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(graph)
        assert system_node.id in preserved

    def test_match_by_importance(self):
        """Test matching by importance."""
        graph = ContextGraph()
        high_importance = graph.add_message(role="user", content="Important")
        high_importance.metadata.importance = 0.95

        low_importance = graph.add_message(role="user", content="Normal")
        low_importance.metadata.importance = 0.3

        pipeline = CompressionPipeline(
            preservation_rules=[
                PreservationRule(name="important", min_importance=0.9),
            ],
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(graph)
        assert high_importance.id in preserved

    def test_match_by_age(self):
        """Test matching by age."""
        graph = ContextGraph()
        recent = graph.add_message(role="user", content="Recent")
        recent.metadata.created_at = datetime.now(UTC)

        old = graph.add_message(role="user", content="Old")
        old.metadata.created_at = datetime.now(UTC) - timedelta(hours=1)

        pipeline = CompressionPipeline(
            preservation_rules=[
                PreservationRule(name="fresh", max_age_seconds=60),
            ],
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(graph)
        assert recent.id in preserved

    def test_match_by_tags(self):
        """Test matching by tags."""
        graph = ContextGraph()
        tagged = graph.add_message(role="user", content="Tagged")
        tagged.metadata.tags = {"important", "keep"}

        graph.add_message(role="user", content="Untagged")

        pipeline = CompressionPipeline(
            preservation_rules=[
                PreservationRule(name="tagged", required_tags={"important"}),
            ],
            config=PipelineConfig(preserve_recent_n=0),
        )

        preserved = pipeline.get_preserved_nodes(graph)
        assert tagged.id in preserved
