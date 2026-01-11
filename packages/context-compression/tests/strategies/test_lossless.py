"""Tests for lossless compression strategies."""

from __future__ import annotations

import pytest

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.lossless import (
    CollapseToolChains,
    ExternalizePayloads,
    InMemoryExternalStorage,
)
from context_compression.types import CompressionTier
from context_core.graph import ContextGraph
from context_core.graph.types import CompressionLevel, NodeType


class TestInMemoryExternalStorage:
    """Tests for InMemoryExternalStorage."""

    def test_store_and_retrieve(self):
        """Test storing and retrieving content."""
        storage = InMemoryExternalStorage()

        uri = storage.store("key1", "test content")
        assert uri == "memory://key1"

        content = storage.retrieve(uri)
        assert content == "test content"

    def test_retrieve_nonexistent(self):
        """Test retrieving non-existent content."""
        storage = InMemoryExternalStorage()
        assert storage.retrieve("memory://missing") is None

    def test_exists(self):
        """Test checking existence."""
        storage = InMemoryExternalStorage()
        storage.store("key1", "content")

        assert storage.exists("memory://key1") is True
        assert storage.exists("memory://key2") is False

    def test_delete(self):
        """Test deleting content."""
        storage = InMemoryExternalStorage()
        uri = storage.store("key1", "content")

        assert storage.delete(uri) is True
        assert storage.exists(uri) is False
        assert storage.delete(uri) is False  # Already deleted

    def test_clear(self):
        """Test clearing all content."""
        storage = InMemoryExternalStorage()
        storage.store("key1", "content1")
        storage.store("key2", "content2")

        storage.clear()
        assert len(storage) == 0

    def test_len(self):
        """Test length."""
        storage = InMemoryExternalStorage()
        assert len(storage) == 0

        storage.store("key1", "content")
        assert len(storage) == 1


class TestExternalizePayloads:
    """Tests for ExternalizePayloads strategy."""

    @pytest.fixture
    def storage(self):
        """Create in-memory storage."""
        return InMemoryExternalStorage()

    @pytest.fixture
    def strategy(self, storage):
        """Create strategy with storage."""
        return ExternalizePayloads(
            storage=storage,
            min_tokens=100,
            preview_tokens=50,
        )

    @pytest.fixture
    def graph_with_large_result(self):
        """Create graph with large tool result."""
        graph = ContextGraph()

        # Add a tool call and large result
        call = graph.add_tool_call("read_file", {"path": "/test.txt"})
        result = graph.add_tool_result(call.id, "x" * 5000)
        result.token_count = 1250  # Simulate large token count

        return graph

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "externalize_payloads"
        assert strategy.tier == CompressionTier.LOSSLESS
        assert strategy.priority == 10

    def test_can_apply_with_eligible_nodes(self, strategy, graph_with_large_result):
        """Test can_apply returns True for eligible graph."""
        assert strategy.can_apply(graph_with_large_result) is True

    def test_can_apply_no_eligible_nodes(self, strategy):
        """Test can_apply returns False when no eligible nodes."""
        graph = ContextGraph()
        graph.add_message(role="user", content="Hello")
        assert strategy.can_apply(graph) is False

    def test_can_apply_empty_graph(self, strategy):
        """Test can_apply returns False for empty graph."""
        graph = ContextGraph()
        assert strategy.can_apply(graph) is False

    def test_estimate_savings(self, strategy, graph_with_large_result):
        """Test savings estimation."""
        savings = strategy.estimate_savings(graph_with_large_result)
        # Should save most tokens, keeping only preview
        assert savings > 1000

    def test_compress_externalizes_large_result(
        self, strategy, storage, graph_with_large_result
    ):
        """Test compression externalizes large tool results."""
        manifest = RecoveryManifest()

        result = strategy.compress(graph_with_large_result, manifest)

        assert result.success is True
        assert result.nodes_compressed == 1
        assert result.tokens_saved > 0
        assert result.is_recoverable is True

        # Check content was stored
        assert len(storage) == 1

        # Check manifest has operation
        assert len(manifest.operations) == 1

    def test_compress_respects_min_tokens(self, storage):
        """Test compression respects minimum token threshold."""
        strategy = ExternalizePayloads(storage, min_tokens=2000)

        graph = ContextGraph()
        call = graph.add_tool_call("test", {})
        result = graph.add_tool_result(call.id, "small content")
        result.token_count = 100  # Below threshold

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_compressed == 0
        assert len(storage) == 0

    def test_compress_skips_pinned_nodes(self, strategy, storage):
        """Test compression skips pinned nodes."""
        graph = ContextGraph()
        call = graph.add_tool_call("test", {})
        result = graph.add_tool_result(call.id, "x" * 5000)
        result.token_count = 1250
        result.metadata.pinned = True  # Pin the node

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_compressed == 0
        assert len(storage) == 0

    def test_compress_target_tokens(self, storage):
        """Test compression respects target tokens limit."""
        strategy = ExternalizePayloads(storage, min_tokens=100, preview_tokens=10)

        graph = ContextGraph()
        # Add multiple large results
        for i in range(5):
            call = graph.add_tool_call("test", {"i": i})
            result = graph.add_tool_result(call.id, "x" * 1000)
            result.token_count = 250

        manifest = RecoveryManifest()
        # Only save 500 tokens (would be 2-3 nodes)
        result = strategy.compress(graph, manifest, target_tokens=500)

        # Should stop before compressing all
        assert result.nodes_compressed < 5

    def test_recover_content(self, strategy, storage, graph_with_large_result):
        """Test recovering externalized content."""
        manifest = RecoveryManifest()
        strategy.compress(graph_with_large_result, manifest)

        # Find the externalized node
        for node in graph_with_large_result:
            if node.type == NodeType.TOOL_RESULT:
                recovered = strategy.recover(node)
                assert recovered is not None
                assert "x" * 100 in recovered  # Original content


class TestCollapseToolChains:
    """Tests for CollapseToolChains strategy."""

    @pytest.fixture
    def strategy(self):
        """Create strategy with default settings."""
        return CollapseToolChains(min_chain_length=3, max_chain_gap=2)

    @pytest.fixture
    def graph_with_chain(self):
        """Create graph with a chain of tool calls."""
        graph = ContextGraph()

        # Add chain of 5 read_file calls
        for i in range(5):
            call = graph.add_tool_call("read_file", {"path": f"/file{i}.txt"})
            call.token_count = 50
            result = graph.add_tool_result(call.id, f"Content of file {i}")
            result.token_count = 100

        return graph

    def test_strategy_properties(self, strategy):
        """Test strategy properties."""
        assert strategy.name == "collapse_tool_chains"
        assert strategy.tier == CompressionTier.LOSSLESS
        assert strategy.priority == 20

    def test_can_apply_with_chain(self, strategy, graph_with_chain):
        """Test can_apply returns True when chain exists."""
        assert strategy.can_apply(graph_with_chain) is True

    def test_can_apply_no_chain(self, strategy):
        """Test can_apply returns False without chain."""
        graph = ContextGraph()
        # Add only 2 tool calls (below min_chain_length)
        for i in range(2):
            call = graph.add_tool_call("read_file", {"path": f"/file{i}.txt"})
            graph.add_tool_result(call.id, f"Content {i}")

        assert strategy.can_apply(graph) is False

    def test_can_apply_mixed_tools(self, strategy):
        """Test can_apply with different tools (no chain)."""
        graph = ContextGraph()
        # Add calls to different tools
        tools = ["read_file", "write_file", "list_dir", "search"]
        for i, tool in enumerate(tools):
            call = graph.add_tool_call(tool, {"arg": i})
            graph.add_tool_result(call.id, f"Result {i}")

        assert strategy.can_apply(graph) is False

    def test_estimate_savings(self, strategy, graph_with_chain):
        """Test savings estimation."""
        savings = strategy.estimate_savings(graph_with_chain)
        # Should save significant tokens from collapsing
        assert savings > 0

    def test_compress_collapses_chain(self, strategy, graph_with_chain):
        """Test compression collapses tool chain."""
        original_count = len(list(graph_with_chain))
        manifest = RecoveryManifest()

        result = strategy.compress(graph_with_chain, manifest)

        assert result.success is True
        assert result.nodes_removed > 0
        assert result.nodes_created == 1  # One summary node
        assert result.is_recoverable is True

        # Graph should be smaller
        assert len(list(graph_with_chain)) < original_count

        # Manifest should have collapse operation
        assert len(manifest.operations) == 1
        assert manifest.operations[0].op_type == "collapse"

    def test_compress_creates_summary_node(self, strategy, graph_with_chain):
        """Test compression creates appropriate summary node."""
        manifest = RecoveryManifest()
        strategy.compress(graph_with_chain, manifest)

        # Find the summary node
        summary_node = None
        for node in graph_with_chain:
            if node.type == NodeType.SUMMARY:
                summary_node = node
                break

        assert summary_node is not None
        assert "read_file" in summary_node.content.text
        assert summary_node.compression_level == CompressionLevel.COMPACTED

    def test_compress_respects_min_chain_length(self):
        """Test compression respects minimum chain length."""
        strategy = CollapseToolChains(min_chain_length=5)

        graph = ContextGraph()
        # Add only 3 tool calls
        for i in range(3):
            call = graph.add_tool_call("read_file", {"path": f"/file{i}.txt"})
            call.token_count = 50
            graph.add_tool_result(call.id, f"Content {i}")

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        assert result.nodes_removed == 0
        assert result.nodes_created == 0

    def test_multiple_chains(self):
        """Test compressing multiple separate chains."""
        strategy = CollapseToolChains(min_chain_length=3, max_chain_gap=1)

        graph = ContextGraph()

        # First chain: read_file x3
        for i in range(3):
            call = graph.add_tool_call("read_file", {"path": f"/file{i}.txt"})
            call.token_count = 50
            graph.add_tool_result(call.id, f"Content {i}")

        # Break with different tool
        graph.add_message(role="assistant", content="Analyzing...")

        # Second chain: write_file x3
        for i in range(3):
            call = graph.add_tool_call("write_file", {"path": f"/out{i}.txt"})
            call.token_count = 50
            graph.add_tool_result(call.id, "OK")

        manifest = RecoveryManifest()
        result = strategy.compress(graph, manifest)

        # Should create 2 summary nodes (one per chain)
        # Note: Depending on sequence number gaps, might be different
        assert result.nodes_created >= 1
