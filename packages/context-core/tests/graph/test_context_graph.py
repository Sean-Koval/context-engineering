"""Tests for ContextGraph class."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from context_core.graph import (
    CompressionLevel,
    Content,
    ContextGraph,
    ContextNode,
    Edge,
    EdgeType,
    NodeMetadata,
    NodeType,
    Role,
)


class TestContextGraphInit:
    """Tests for ContextGraph initialization."""

    def test_default_init(self) -> None:
        """Graph initializes with defaults."""
        g = ContextGraph()
        assert len(g) == 0
        assert isinstance(g.session_id, str)
        assert len(g.session_id) == 36  # UUID format

    def test_custom_session_id(self) -> None:
        """Can provide custom session ID."""
        g = ContextGraph(session_id="my-session")
        assert g.session_id == "my-session"


class TestContextGraphCRUD:
    """Tests for basic CRUD operations."""

    def test_add_node(self) -> None:
        """Can add a node to the graph."""
        g = ContextGraph()
        node = ContextNode(type=NodeType.MESSAGE, content=Content(text="Hello"))

        result = g.add_node(node)

        assert result is node
        assert result.sequence_number == 0
        assert len(g) == 1
        assert node.id in g

    def test_add_multiple_nodes(self) -> None:
        """Nodes get sequential sequence numbers."""
        g = ContextGraph()

        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="First"))
        )
        n2 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Second"))
        )
        n3 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Third"))
        )

        assert n1.sequence_number == 0
        assert n2.sequence_number == 1
        assert n3.sequence_number == 2
        assert len(g) == 3

    def test_temporal_edges_auto_created(self) -> None:
        """TEMPORAL edges are created between consecutive nodes."""
        g = ContextGraph()

        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="First"))
        )
        n2 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Second"))
        )

        edges = g.get_edges(source_id=n1.id)
        assert len(edges) == 1
        assert edges[0].type == EdgeType.TEMPORAL
        assert edges[0].target_id == n2.id

    def test_disable_temporal_edges(self) -> None:
        """Can disable automatic temporal edges."""
        g = ContextGraph()

        g.add_node(ContextNode(type=NodeType.MESSAGE, content=Content(text="First")))
        g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Second")),
            connect_temporal=False,
        )

        edges = g.get_edges()
        assert len(edges) == 0

    def test_explicit_predecessor(self) -> None:
        """Can specify explicit predecessor for temporal edge."""
        g = ContextGraph()

        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="First"))
        )
        g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Second")),
            connect_temporal=False,
        )
        n3 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Third")),
            predecessor_id=n1.id,  # Skip n2
        )

        edges = g.get_edges(target_id=n3.id)
        assert len(edges) == 1
        assert edges[0].source_id == n1.id

    def test_get_node(self) -> None:
        """Can retrieve a node by ID."""
        g = ContextGraph()
        node = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Test"))
        )

        result = g.get_node(node.id)
        assert result is node

    def test_get_node_not_found(self) -> None:
        """Returns None for non-existent node."""
        g = ContextGraph()
        result = g.get_node(uuid4())
        assert result is None

    def test_update_node(self) -> None:
        """Can update node fields."""
        g = ContextGraph()
        node = g.add_node(
            ContextNode(
                type=NodeType.MESSAGE,
                content=Content(text="Original"),
                metadata=NodeMetadata(importance=0.5),
            )
        )

        result = g.update_node(node.id, compression_level=CompressionLevel.COMPACTED)
        assert result is node
        assert node.compression_level == CompressionLevel.COMPACTED

    def test_update_node_metadata(self) -> None:
        """Can update metadata fields via update_node."""
        g = ContextGraph()
        node = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Test"))
        )

        g.update_node(node.id, importance=0.9)
        assert node.metadata.importance == 0.9

    def test_update_node_content(self) -> None:
        """Can update content fields via update_node."""
        g = ContextGraph()
        node = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Old text"))
        )

        g.update_node(node.id, text="New text")
        assert node.content.text == "New text"

    def test_update_node_not_found(self) -> None:
        """update_node returns None for non-existent node."""
        g = ContextGraph()
        result = g.update_node(uuid4(), importance=0.9)
        assert result is None

    def test_remove_node(self) -> None:
        """Can remove a node."""
        g = ContextGraph()
        node = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="Test"))
        )

        result = g.remove_node(node.id)
        assert result is node
        assert len(g) == 0
        assert node.id not in g

    def test_remove_node_removes_edges(self) -> None:
        """Removing a node also removes its edges."""
        g = ContextGraph()
        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="First"))
        )
        g.add_node(ContextNode(type=NodeType.MESSAGE, content=Content(text="Second")))

        g.remove_node(n1.id)

        edges = g.get_edges()
        assert len(edges) == 0

    def test_remove_node_not_found(self) -> None:
        """remove_node returns None for non-existent node."""
        g = ContextGraph()
        result = g.remove_node(uuid4())
        assert result is None


class TestContextGraphConvenienceMethods:
    """Tests for add_message, add_tool_call, add_tool_result."""

    def test_add_message(self) -> None:
        """add_message creates proper node."""
        g = ContextGraph()
        node = g.add_message("user", "Hello!")

        assert node.type == NodeType.MESSAGE
        assert node.content.text == "Hello!"
        assert node.content.role == Role.USER

    def test_add_message_with_metadata(self) -> None:
        """add_message accepts metadata kwargs."""
        g = ContextGraph()
        node = g.add_message("assistant", "Response", importance=0.9)

        assert node.metadata.importance == 0.9

    def test_add_tool_call(self) -> None:
        """add_tool_call creates proper node."""
        g = ContextGraph()
        node = g.add_tool_call("search", {"query": "test", "limit": 5})

        assert node.type == NodeType.TOOL_CALL
        assert node.content.tool_name == "search"
        assert node.content.tool_args == {"query": "test", "limit": 5}

    def test_add_tool_result(self) -> None:
        """add_tool_result creates node and TOOL_IO edge."""
        g = ContextGraph()
        call = g.add_tool_call("search", {"query": "test"})
        result = g.add_tool_result(call.id, {"results": [1, 2, 3]})

        assert result.type == NodeType.TOOL_RESULT
        assert result.content.tool_output == {"results": [1, 2, 3]}
        assert result.content.is_error is False

        # Check TOOL_IO edge
        edges = g.get_edges(source_id=call.id, edge_type=EdgeType.TOOL_IO)
        assert len(edges) == 1
        assert edges[0].target_id == result.id

    def test_add_tool_result_error(self) -> None:
        """add_tool_result can indicate error."""
        g = ContextGraph()
        call = g.add_tool_call("api", {"url": "http://fail"})
        result = g.add_tool_result(call.id, "Connection refused", is_error=True)

        assert result.content.is_error is True
        assert result.content.tool_output == "Connection refused"


class TestContextGraphEdges:
    """Tests for edge operations."""

    def test_add_edge(self) -> None:
        """Can manually add an edge."""
        g = ContextGraph()
        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="A")),
            connect_temporal=False,
        )
        n2 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="B")),
            connect_temporal=False,
        )

        edge = g.add_edge(
            Edge(source_id=n1.id, target_id=n2.id, type=EdgeType.REFERENCES)
        )

        assert edge.type == EdgeType.REFERENCES
        edges = g.get_edges()
        assert len(edges) == 1

    def test_add_edge_requires_existing_nodes(self) -> None:
        """add_edge fails if nodes don't exist."""
        g = ContextGraph()
        n1 = g.add_node(
            ContextNode(type=NodeType.MESSAGE, content=Content(text="A")),
            connect_temporal=False,
        )

        with pytest.raises(ValueError, match="must exist"):
            g.add_edge(Edge(source_id=n1.id, target_id=uuid4(), type=EdgeType.CAUSAL))

    def test_get_edges_by_source(self) -> None:
        """Can filter edges by source."""
        g = ContextGraph()
        n1 = g.add_message("user", "A")
        n2 = g.add_message("assistant", "B")
        g.add_message("user", "C")

        edges = g.get_edges(source_id=n1.id)
        assert len(edges) == 1
        assert edges[0].target_id == n2.id

    def test_get_edges_by_target(self) -> None:
        """Can filter edges by target."""
        g = ContextGraph()
        n1 = g.add_message("user", "A")
        n2 = g.add_message("assistant", "B")

        edges = g.get_edges(target_id=n2.id)
        assert len(edges) == 1
        assert edges[0].source_id == n1.id

    def test_get_edges_by_type(self) -> None:
        """Can filter edges by type."""
        g = ContextGraph()
        call = g.add_tool_call("search", {})
        g.add_tool_result(call.id, {})

        temporal = g.get_edges(edge_type=EdgeType.TEMPORAL)
        tool_io = g.get_edges(edge_type=EdgeType.TOOL_IO)

        assert len(temporal) == 1
        assert len(tool_io) == 1


class TestContextGraphQueries:
    """Tests for query operations."""

    def test_query_by_type(self) -> None:
        """Can query nodes by type."""
        g = ContextGraph()
        g.add_message("user", "Hello")
        g.add_tool_call("search", {})
        g.add_message("assistant", "Response")

        messages = g.query_nodes(node_types=[NodeType.MESSAGE])
        assert len(messages) == 2

        tools = g.query_nodes(node_types=[NodeType.TOOL_CALL])
        assert len(tools) == 1

    def test_query_by_importance(self) -> None:
        """Can filter by minimum importance."""
        g = ContextGraph()
        g.add_message("user", "Low", importance=0.1)
        g.add_message("user", "High", importance=0.9)

        high = g.query_nodes(min_importance=0.5)
        assert len(high) == 1

    def test_query_by_compression_level(self) -> None:
        """Can filter by maximum compression level."""
        g = ContextGraph()
        g.add_message("user", "Full")
        n2 = g.add_message("user", "Compacted")
        n2.compression_level = CompressionLevel.COMPACTED
        n3 = g.add_message("user", "Summarized")
        n3.compression_level = CompressionLevel.SUMMARIZED

        full_only = g.query_nodes(max_compression_level=CompressionLevel.FULL)
        assert len(full_only) == 1

        up_to_compacted = g.query_nodes(
            max_compression_level=CompressionLevel.COMPACTED
        )
        assert len(up_to_compacted) == 2

    def test_query_by_tags(self) -> None:
        """Can filter by tags."""
        g = ContextGraph()
        n1 = g.add_message("user", "Tagged")
        n1.metadata.tags = {"important", "reviewed"}
        g.add_message("user", "Untagged")

        result = g.query_nodes(tags={"important"})
        assert len(result) == 1
        assert result[0].id == n1.id

    def test_query_by_entity(self) -> None:
        """Can filter by entity references."""
        g = ContextGraph()
        n1 = g.add_message("user", "About John")
        n1.metadata.entities = ["person:john"]
        n2 = g.add_message("user", "About Jane")
        n2.metadata.entities = ["person:jane"]

        result = g.query_nodes(entity_ids=["person:john"])
        assert len(result) == 1
        assert result[0].id == n1.id

    def test_query_by_time_range(self) -> None:
        """Can filter by creation time."""
        g = ContextGraph()
        now = datetime.now(UTC)

        n1 = g.add_message("user", "Old")
        n1.metadata.created_at = now - timedelta(hours=2)
        n2 = g.add_message("user", "Recent")
        n2.metadata.created_at = now - timedelta(minutes=30)

        result = g.query_nodes(since=now - timedelta(hours=1))
        assert len(result) == 1
        assert result[0].id == n2.id

    def test_query_ordering_by_sequence(self) -> None:
        """Nodes are ordered by sequence by default."""
        g = ContextGraph()
        g.add_message("user", "First")
        g.add_message("user", "Second")
        g.add_message("user", "Third")

        result = g.query_nodes()
        assert [n.content.text for n in result] == ["First", "Second", "Third"]

    def test_query_ordering_by_importance(self) -> None:
        """Can order by importance (descending)."""
        g = ContextGraph()
        g.add_message("user", "Low", importance=0.2)
        g.add_message("user", "High", importance=0.9)
        g.add_message("user", "Medium", importance=0.5)

        result = g.query_nodes(order_by="importance")
        texts = [n.content.text for n in result]
        assert texts[0] == "High"
        assert texts[-1] == "Low"

    def test_query_with_limit(self) -> None:
        """Can limit query results."""
        g = ContextGraph()
        for i in range(10):
            g.add_message("user", f"Message {i}")

        result = g.query_nodes(limit=3)
        assert len(result) == 3

    def test_get_recent(self) -> None:
        """get_recent returns N most recent nodes."""
        g = ContextGraph()
        for i in range(10):
            g.add_message("user", f"Message {i}")

        result = g.get_recent(3)
        assert len(result) == 3
        assert result[0].content.text == "Message 7"
        assert result[-1].content.text == "Message 9"

    def test_get_by_entity(self) -> None:
        """get_by_entity is a convenience wrapper."""
        g = ContextGraph()
        n1 = g.add_message("user", "About project X")
        n1.metadata.entities = ["project:x"]

        result = g.get_by_entity("project:x")
        assert len(result) == 1

    def test_get_neighbors(self) -> None:
        """get_neighbors returns connected nodes."""
        g = ContextGraph()
        n1 = g.add_message("user", "First")
        n2 = g.add_message("assistant", "Second")
        n3 = g.add_message("user", "Third")

        # n2 has n1 as predecessor and n3 as successor (TEMPORAL edges)
        neighbors = g.get_neighbors(n2.id)
        assert len(neighbors) == 2

        out_neighbors = g.get_neighbors(n2.id, direction="out")
        assert len(out_neighbors) == 1
        assert out_neighbors[0].id == n3.id

        in_neighbors = g.get_neighbors(n2.id, direction="in")
        assert len(in_neighbors) == 1
        assert in_neighbors[0].id == n1.id

    def test_get_neighbors_by_edge_type(self) -> None:
        """Can filter neighbors by edge type."""
        g = ContextGraph()
        call = g.add_tool_call("search", {})
        g.add_tool_result(call.id, {})

        # call has temporal edge to result AND TOOL_IO edge
        temporal_neighbors = g.get_neighbors(call.id, edge_types=[EdgeType.TEMPORAL])
        assert len(temporal_neighbors) == 1

        tool_neighbors = g.get_neighbors(call.id, edge_types=[EdgeType.TOOL_IO])
        assert len(tool_neighbors) == 1


class TestContextGraphSubgraph:
    """Tests for subgraph extraction."""

    def test_get_subgraph_basic(self) -> None:
        """Can extract a subgraph."""
        g = ContextGraph()
        g.add_message("user", "First")
        n2 = g.add_message("assistant", "Second")
        g.add_message("user", "Third")

        sub = g.get_subgraph([n2.id], max_depth=1)

        assert len(sub) == 3  # n2 plus both neighbors

    def test_get_subgraph_depth_limit(self) -> None:
        """Subgraph respects depth limit."""
        g = ContextGraph()
        n1 = g.add_message("user", "First")
        n2 = g.add_message("assistant", "Second")
        n3 = g.add_message("user", "Third")
        g.add_message("assistant", "Fourth")

        sub = g.get_subgraph([n1.id], max_depth=1)

        assert n1.id in sub
        assert n2.id in sub
        assert n3.id not in sub  # Depth 2 from n1

    def test_get_subgraph_edge_filter(self) -> None:
        """Can filter subgraph by edge type."""
        g = ContextGraph()
        call = g.add_tool_call("search", {})
        g.add_tool_result(call.id, {})

        # Only follow TOOL_IO edges
        sub = g.get_subgraph([call.id], max_depth=1, edge_types=[EdgeType.TOOL_IO])

        assert len(sub) == 2  # call and result

    def test_get_subgraph_preserves_edges(self) -> None:
        """Subgraph includes edges between extracted nodes."""
        g = ContextGraph()
        n1 = g.add_message("user", "First")
        g.add_message("assistant", "Second")

        sub = g.get_subgraph([n1.id], max_depth=1)
        edges = sub.get_edges()

        assert len(edges) == 1
        assert edges[0].type == EdgeType.TEMPORAL


class TestContextGraphSerialization:
    """Tests for to_dict/from_dict and to_messages."""

    def test_to_dict(self) -> None:
        """Graph serializes to dictionary."""
        g = ContextGraph(session_id="test-session")
        g.add_message("user", "Hello")
        g.add_message("assistant", "Hi")

        data = g.to_dict()

        assert data["session_id"] == "test-session"
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_from_dict(self) -> None:
        """Graph deserializes from dictionary."""
        g1 = ContextGraph(session_id="test-session")
        n1 = g1.add_message("user", "Hello")
        n2 = g1.add_message("assistant", "Hi")

        data = g1.to_dict()
        g2 = ContextGraph.from_dict(data)

        assert g2.session_id == "test-session"
        assert len(g2) == 2
        assert g2.get_node(n1.id) is not None
        assert g2.get_node(n2.id) is not None

    def test_roundtrip_preserves_data(self) -> None:
        """Serialization roundtrip preserves all data."""
        g1 = ContextGraph()
        call = g1.add_tool_call("search", {"q": "test"})
        g1.add_tool_result(call.id, {"results": []})

        data = g1.to_dict()
        g2 = ContextGraph.from_dict(data)

        # Check tool call
        tc = g2.get_node(call.id)
        assert tc is not None
        assert tc.content.tool_name == "search"
        assert tc.content.tool_args == {"q": "test"}

        # Check edges
        edges = g2.get_edges(edge_type=EdgeType.TOOL_IO)
        assert len(edges) == 1

    def test_to_messages(self) -> None:
        """Graph converts to LLM message list."""
        g = ContextGraph()
        g.add_message("user", "Hello")
        g.add_message("assistant", "Hi there!")

        messages = g.to_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there!"}

    def test_to_messages_includes_tool_calls(self) -> None:
        """to_messages includes tool calls and results."""
        g = ContextGraph()
        g.add_message("user", "Search for X")
        g.add_tool_call("search", {"q": "X"})
        g.add_tool_result(g.get_recent(1)[0].id, {"found": True})
        g.add_message("assistant", "Found it")

        messages = g.to_messages()
        assert len(messages) == 4

    def test_to_messages_exclude_compressed(self) -> None:
        """to_messages can exclude compressed nodes."""
        g = ContextGraph()
        g.add_message("user", "Keep me")
        n2 = g.add_message("user", "Compress me")
        n2.compression_level = CompressionLevel.SUMMARIZED

        messages = g.to_messages(exclude_compressed=True)
        assert len(messages) == 1
        assert messages[0]["content"] == "Keep me"


class TestContextGraphStats:
    """Tests for statistics."""

    def test_stats_empty_graph(self) -> None:
        """Stats work on empty graph."""
        g = ContextGraph()
        stats = g.stats()

        assert stats.node_count == 0
        assert stats.edge_count == 0
        assert stats.total_tokens == 0
        assert stats.avg_importance == 0.0

    def test_stats_with_nodes(self) -> None:
        """Stats count nodes and edges."""
        g = ContextGraph()
        g.add_message("user", "Hello")
        g.add_message("assistant", "Hi")

        stats = g.stats()

        assert stats.node_count == 2
        assert stats.edge_count == 1
        assert stats.nodes_by_type == {"message": 2}
        assert stats.nodes_by_compression == {"FULL": 2}

    def test_stats_by_type(self) -> None:
        """Stats group by node type."""
        g = ContextGraph()
        g.add_message("user", "Hello")
        call = g.add_tool_call("search", {})
        g.add_tool_result(call.id, {})

        stats = g.stats()

        assert stats.nodes_by_type["message"] == 1
        assert stats.nodes_by_type["tool_call"] == 1
        assert stats.nodes_by_type["tool_result"] == 1

    def test_stats_token_count(self) -> None:
        """Stats sum token counts."""
        g = ContextGraph()
        n1 = g.add_message("user", "Hello")
        n1.token_count = 10
        n2 = g.add_message("assistant", "Hi")
        n2.token_count = 5

        stats = g.stats()
        assert stats.total_tokens == 15


class TestContextGraphIteration:
    """Tests for iteration protocol."""

    def test_iter(self) -> None:
        """Can iterate over graph nodes."""
        g = ContextGraph()
        g.add_message("user", "First")
        g.add_message("assistant", "Second")
        g.add_message("user", "Third")

        nodes = list(g)
        assert len(nodes) == 3
        assert nodes[0].content.text == "First"
        assert nodes[2].content.text == "Third"

    def test_len(self) -> None:
        """len() returns node count."""
        g = ContextGraph()
        assert len(g) == 0

        g.add_message("user", "Test")
        assert len(g) == 1

    def test_contains(self) -> None:
        """'in' operator checks node existence."""
        g = ContextGraph()
        node = g.add_message("user", "Test")

        assert node.id in g
        assert uuid4() not in g
