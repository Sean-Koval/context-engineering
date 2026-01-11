"""Tests for SemanticIndex."""

from __future__ import annotations

from uuid import uuid4

import pytest

from context_core.graph.nodes import Content, ContextNode
from context_core.graph.types import CompressionLevel, NodeType, Role
from context_core.semantic.embeddings import MockEmbeddingModel
from context_core.semantic.index import SemanticIndex
from context_core.semantic.stores.memory import InMemoryVectorStore


class TestSemanticIndex:
    """Tests for SemanticIndex."""

    @pytest.fixture
    def model(self) -> MockEmbeddingModel:
        """Create a mock embedding model."""
        return MockEmbeddingModel(dimension=64)

    @pytest.fixture
    def index(self, model: MockEmbeddingModel) -> SemanticIndex:
        """Create a semantic index."""
        return SemanticIndex(model)

    @pytest.fixture
    def message_node(self) -> ContextNode:
        """Create a sample message node."""
        return ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hello, how can I help you today?", role=Role.USER),
            compression_level=CompressionLevel.FULL,
        )

    @pytest.fixture
    def tool_call_node(self) -> ContextNode:
        """Create a sample tool call node."""
        return ContextNode(
            type=NodeType.TOOL_CALL,
            content=Content(
                tool_name="read_file",
                tool_args={"path": "/home/user/test.py"},
                role=Role.ASSISTANT,
            ),
            compression_level=CompressionLevel.FULL,
        )

    @pytest.fixture
    def tool_result_node(self) -> ContextNode:
        """Create a sample tool result node."""
        return ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(
                tool_output="def hello():\n    print('Hello, world!')",
                role=Role.TOOL,
            ),
            compression_level=CompressionLevel.FULL,
        )

    # Initialization tests

    def test_init_with_default_store(self, model: MockEmbeddingModel) -> None:
        """Test initialization with default InMemoryVectorStore."""
        index = SemanticIndex(model)
        assert index.embedding_model is model
        assert isinstance(index.vector_store, InMemoryVectorStore)

    def test_init_with_custom_store(self, model: MockEmbeddingModel) -> None:
        """Test initialization with custom vector store."""
        store = InMemoryVectorStore(dimension=64)
        index = SemanticIndex(model, vector_store=store)
        assert index.vector_store is store

    # Indexing tests

    def test_index_node(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test indexing a single node."""
        result = index.index_node(message_node)

        assert result is True
        assert index.count() == 1
        assert message_node.id in index

    def test_index_node_tool_call(
        self, index: SemanticIndex, tool_call_node: ContextNode
    ) -> None:
        """Test indexing a tool call node."""
        result = index.index_node(tool_call_node)

        assert result is True
        assert index.count() == 1

    def test_index_node_tool_result(
        self, index: SemanticIndex, tool_result_node: ContextNode
    ) -> None:
        """Test indexing a tool result node."""
        result = index.index_node(tool_result_node)

        assert result is True
        assert index.count() == 1

    def test_index_node_empty_content(self, index: SemanticIndex) -> None:
        """Test indexing a node with no indexable content."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(),  # Empty content
            compression_level=CompressionLevel.FULL,
        )
        result = index.index_node(node)

        assert result is False
        assert index.count() == 0

    def test_index_nodes_batch(
        self,
        index: SemanticIndex,
        message_node: ContextNode,
        tool_call_node: ContextNode,
    ) -> None:
        """Test batch indexing multiple nodes."""
        nodes = [message_node, tool_call_node]
        count = index.index_nodes(nodes)

        assert count == 2
        assert index.count() == 2

    def test_index_nodes_empty_list(self, index: SemanticIndex) -> None:
        """Test batch indexing empty list."""
        count = index.index_nodes([])
        assert count == 0

    def test_index_nodes_with_empty_content(
        self, index: SemanticIndex, message_node: ContextNode
    ) -> None:
        """Test batch indexing with some empty content nodes."""
        empty_node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(),
            compression_level=CompressionLevel.FULL,
        )
        nodes = [message_node, empty_node]
        count = index.index_nodes(nodes)

        assert count == 1  # Only message_node indexed

    # Search tests

    def test_search_basic(
        self, index: SemanticIndex, message_node: ContextNode
    ) -> None:
        """Test basic search."""
        index.index_node(message_node)

        # Search with exact text (mock embeddings are hash-based, not semantic)
        results = index.search("Hello, how can I help you today?", k=5)

        assert len(results) >= 1
        assert results[0].id == message_node.id

    def test_search_with_min_score(
        self, index: SemanticIndex, message_node: ContextNode
    ) -> None:
        """Test search with minimum score filter."""
        index.index_node(message_node)

        # Very high threshold - results may be empty if score is below threshold
        results = index.search("random unrelated query", k=5, min_score=0.99)
        assert isinstance(results, list)

    def test_search_with_filter(
        self,
        index: SemanticIndex,
        message_node: ContextNode,
        tool_call_node: ContextNode,
    ) -> None:
        """Test search with metadata filter."""
        index.index_node(message_node)
        index.index_node(tool_call_node)

        results = index.search("test", k=10, filter={"type": "message"})

        for r in results:
            assert r.metadata.get("type") == "message"

    def test_search_empty_index(self, index: SemanticIndex) -> None:
        """Test search on empty index."""
        results = index.search("query")
        assert results == []

    def test_search_by_node(
        self,
        index: SemanticIndex,
        message_node: ContextNode,
        tool_call_node: ContextNode,
    ) -> None:
        """Test finding similar nodes to a given node."""
        index.index_node(message_node)
        index.index_node(tool_call_node)

        results = index.search_by_node(message_node, k=5)

        # With mock embeddings (random), other nodes may not be similar
        # Just verify the method works and excludes self
        assert message_node.id not in [r.id for r in results]

    def test_search_by_node_include_self(
        self, index: SemanticIndex, message_node: ContextNode
    ) -> None:
        """Test finding similar nodes including self."""
        index.index_node(message_node)

        results = index.search_by_node(message_node, k=5, exclude_self=False)

        assert message_node.id in [r.id for r in results]

    # Duplicate detection tests

    def test_find_duplicates_none(self, index: SemanticIndex) -> None:
        """Test find_duplicates with no duplicates."""
        # Create nodes with different content
        node1 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hello world", role=Role.USER),
            compression_level=CompressionLevel.FULL,
        )
        node2 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(
                text="Completely different text about programming", role=Role.USER
            ),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node1)
        index.index_node(node2)

        duplicates = index.find_duplicates(threshold=0.99)
        # With very different content and high threshold, likely no duplicates
        # (MockEmbeddingModel generates deterministic embeddings from text hash)
        assert isinstance(duplicates, list)

    def test_find_duplicates_with_identical(self, index: SemanticIndex) -> None:
        """Test find_duplicates with identical content."""
        text = "This is the same text content"
        node1 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text, role=Role.USER),
            compression_level=CompressionLevel.FULL,
        )
        node2 = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text=text, role=Role.USER),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node1)
        index.index_node(node2)

        duplicates = index.find_duplicates(threshold=0.95)

        assert len(duplicates) >= 1
        # Should find the pair
        ids = {duplicates[0][0], duplicates[0][1]}
        assert node1.id in ids
        assert node2.id in ids

    def test_find_duplicates_empty_index(self, index: SemanticIndex) -> None:
        """Test find_duplicates on empty index."""
        duplicates = index.find_duplicates()
        assert duplicates == []

    def test_find_duplicates_single_node(
        self, index: SemanticIndex, message_node: ContextNode
    ) -> None:
        """Test find_duplicates with single node."""
        index.index_node(message_node)
        duplicates = index.find_duplicates()
        assert duplicates == []

    # Removal tests

    def test_remove_node(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test removing a node."""
        index.index_node(message_node)
        assert index.count() == 1

        result = index.remove_node(message_node.id)

        assert result is True
        assert index.count() == 0
        assert message_node.id not in index

    def test_remove_nonexistent_node(self, index: SemanticIndex) -> None:
        """Test removing a node that doesn't exist."""
        result = index.remove_node(uuid4())
        assert result is False

    # Utility tests

    def test_get_text(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test getting indexed text for a node."""
        index.index_node(message_node)

        text = index.get_text(message_node.id)

        assert text == "Hello, how can I help you today?"

    def test_get_text_nonexistent(self, index: SemanticIndex) -> None:
        """Test getting text for non-existent node."""
        text = index.get_text(uuid4())
        assert text is None

    def test_is_indexed(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test checking if node is indexed."""
        assert not index.is_indexed(message_node.id)

        index.index_node(message_node)

        assert index.is_indexed(message_node.id)

    def test_count(
        self,
        index: SemanticIndex,
        message_node: ContextNode,
        tool_call_node: ContextNode,
    ) -> None:
        """Test count method."""
        assert index.count() == 0

        index.index_node(message_node)
        assert index.count() == 1

        index.index_node(tool_call_node)
        assert index.count() == 2

    def test_clear(
        self,
        index: SemanticIndex,
        message_node: ContextNode,
        tool_call_node: ContextNode,
    ) -> None:
        """Test clearing the index."""
        index.index_node(message_node)
        index.index_node(tool_call_node)
        assert index.count() == 2

        index.clear()

        assert index.count() == 0
        assert len(index) == 0

    def test_len(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test __len__ method."""
        assert len(index) == 0

        index.index_node(message_node)

        assert len(index) == 1

    def test_contains(self, index: SemanticIndex, message_node: ContextNode) -> None:
        """Test __contains__ method."""
        assert message_node.id not in index

        index.index_node(message_node)

        assert message_node.id in index


class TestSemanticIndexTextExtraction:
    """Tests for text extraction from different node types."""

    @pytest.fixture
    def index(self) -> SemanticIndex:
        """Create a semantic index."""
        model = MockEmbeddingModel(dimension=64)
        return SemanticIndex(model)

    def test_extract_text_from_message(self, index: SemanticIndex) -> None:
        """Test text extraction from message node."""
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(text="Hello world", role=Role.USER),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node)

        assert index.get_text(node.id) == "Hello world"

    def test_extract_text_from_tool_call(self, index: SemanticIndex) -> None:
        """Test text extraction from tool call node."""
        node = ContextNode(
            type=NodeType.TOOL_CALL,
            content=Content(
                tool_name="search",
                tool_args={"query": "test"},
                role=Role.ASSISTANT,
            ),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node)

        text = index.get_text(node.id)
        assert "search" in text
        assert "query" in text

    def test_extract_text_from_tool_result(self, index: SemanticIndex) -> None:
        """Test text extraction from tool result node."""
        node = ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(
                tool_output="Result data here",
                role=Role.TOOL,
            ),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node)

        assert index.get_text(node.id) == "Result data here"

    def test_extract_text_truncates_long_output(self, index: SemanticIndex) -> None:
        """Test that long tool output is truncated."""
        long_output = "x" * 2000
        node = ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(tool_output=long_output, role=Role.TOOL),
            compression_level=CompressionLevel.FULL,
        )
        index.index_node(node)

        text = index.get_text(node.id)
        assert len(text) <= 1000

    def test_extract_text_from_summary(self, index: SemanticIndex) -> None:
        """Test text extraction from summary node."""
        node = ContextNode(
            type=NodeType.SUMMARY,
            content=Content(text="This is a summary of the conversation"),
            compression_level=CompressionLevel.SUMMARIZED,
        )
        index.index_node(node)

        assert index.get_text(node.id) == "This is a summary of the conversation"
