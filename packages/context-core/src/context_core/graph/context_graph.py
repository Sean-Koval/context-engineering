"""ContextGraph - Graph-based context representation.

This module provides the main ContextGraph class which manages
typed nodes and edges representing LLM conversation context.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import networkx as nx
from pydantic import BaseModel

from context_core.graph.edges import Edge, EdgeMetadata
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, EdgeType, NodeType, Role

if TYPE_CHECKING:
    from context_core.tokenizer import Tokenizer


class GraphStats(BaseModel):
    """Statistics about the context graph.

    Attributes:
        node_count: Total number of nodes in the graph
        edge_count: Total number of edges in the graph
        total_tokens: Sum of token counts across all nodes
        nodes_by_type: Count of nodes grouped by NodeType
        nodes_by_compression: Count of nodes grouped by CompressionLevel
        avg_importance: Average importance score across all nodes
    """

    node_count: int
    edge_count: int
    total_tokens: int
    nodes_by_type: dict[str, int]
    nodes_by_compression: dict[str, int]
    avg_importance: float


class ContextGraph:
    """Graph-based context representation with typed nodes and edges.

    The ContextGraph is the central data structure for managing LLM
    conversation context. It stores nodes (messages, tool calls, etc.)
    and edges (temporal, causal, reference relationships) in a directed
    graph structure.

    Thread-safe for read operations. Write operations should be
    externally synchronized in multi-threaded environments.

    Attributes:
        session_id: Unique identifier for this context session

    Example:
        >>> graph = ContextGraph()
        >>> msg = graph.add_message("user", "Hello!")
        >>> tool = graph.add_tool_call("search", {"query": "test"})
        >>> result = graph.add_tool_result(tool.id, {"results": []})
    """

    def __init__(self, session_id: str | None = None) -> None:
        """Initialize a new ContextGraph.

        Args:
            session_id: Optional session identifier. If not provided,
                a new UUID will be generated.
        """
        # Use MultiDiGraph to support multiple edges between same node pair
        # (e.g., both TEMPORAL and TOOL_IO edges from tool_call to tool_result)
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: dict[UUID, ContextNode] = {}
        self._sequence_counter: int = 0
        self.session_id: str = session_id or str(uuid4())
        self._created_at: datetime = datetime.now(UTC)

    # === CRUD Operations ===

    def add_node(
        self,
        node: ContextNode,
        *,
        connect_temporal: bool = True,
        predecessor_id: UUID | None = None,
    ) -> ContextNode:
        """Add a node to the graph.

        Args:
            node: The node to add
            connect_temporal: If True, add TEMPORAL edge from last node
            predecessor_id: Explicit predecessor for TEMPORAL edge

        Returns:
            The added node (with sequence_number set)
        """
        node.sequence_number = self._sequence_counter
        self._sequence_counter += 1

        self._nodes[node.id] = node
        self._graph.add_node(node.id)

        if connect_temporal and len(self._nodes) > 1:
            if predecessor_id:
                pred = predecessor_id
            else:
                # Find the node with highest sequence number (excluding current)
                pred = max(
                    (n for n in self._nodes.values() if n.id != node.id),
                    key=lambda n: n.sequence_number or 0,
                ).id

            self.add_edge(
                Edge(
                    source_id=pred,
                    target_id=node.id,
                    type=EdgeType.TEMPORAL,
                )
            )

        return node

    def add_message(
        self,
        role: str,
        content: str,
        **metadata_kwargs: Any,
    ) -> ContextNode:
        """Convenience method to add a message node.

        Args:
            role: The message role ("user", "assistant", "system")
            content: The message text content
            **metadata_kwargs: Additional NodeMetadata fields

        Returns:
            The created and added ContextNode
        """
        node = ContextNode(
            type=NodeType.MESSAGE,
            content=Content(
                text=content,
                role=Role(role),
            ),
            metadata=NodeMetadata(**metadata_kwargs),
        )
        return self.add_node(node)

    def add_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        **metadata_kwargs: Any,
    ) -> ContextNode:
        """Convenience method to add a tool call node.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool
            **metadata_kwargs: Additional NodeMetadata fields

        Returns:
            The created and added ContextNode
        """
        node = ContextNode(
            type=NodeType.TOOL_CALL,
            content=Content(
                tool_name=tool_name,
                tool_args=tool_args,
            ),
            metadata=NodeMetadata(**metadata_kwargs),
        )
        return self.add_node(node)

    def add_tool_result(
        self,
        tool_call_id: UUID,
        output: Any,
        is_error: bool = False,
        **metadata_kwargs: Any,
    ) -> ContextNode:
        """Add a tool result and link it to the tool call.

        Args:
            tool_call_id: UUID of the corresponding tool call node
            output: The result from tool execution
            is_error: Whether the tool execution failed
            **metadata_kwargs: Additional NodeMetadata fields

        Returns:
            The created and added ContextNode
        """
        node = ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(
                tool_call_id=tool_call_id,
                tool_output=output,
                is_error=is_error,
            ),
            metadata=NodeMetadata(**metadata_kwargs),
        )
        result = self.add_node(node)

        # Add TOOL_IO edge from call to result
        self.add_edge(
            Edge(
                source_id=tool_call_id,
                target_id=result.id,
                type=EdgeType.TOOL_IO,
            )
        )

        return result

    def get_node(self, node_id: UUID) -> ContextNode | None:
        """Get a node by ID.

        Args:
            node_id: The UUID of the node to retrieve

        Returns:
            The ContextNode if found, None otherwise
        """
        return self._nodes.get(node_id)

    def update_node(self, node_id: UUID, **updates: Any) -> ContextNode | None:
        """Update node fields.

        Updates are applied to the node, its content, or its metadata
        depending on which object has the attribute.

        Args:
            node_id: The UUID of the node to update
            **updates: Field name/value pairs to update

        Returns:
            Updated node or None if not found
        """
        node = self._nodes.get(node_id)
        if not node:
            return None

        for key, value in updates.items():
            if hasattr(node, key):
                setattr(node, key, value)
            elif hasattr(node.metadata, key):
                setattr(node.metadata, key, value)
            elif hasattr(node.content, key):
                setattr(node.content, key, value)

        return node

    def remove_node(self, node_id: UUID) -> ContextNode | None:
        """Remove a node and its edges.

        Args:
            node_id: The UUID of the node to remove

        Returns:
            The removed node or None if not found
        """
        node = self._nodes.pop(node_id, None)
        if node and node_id in self._graph:
            self._graph.remove_node(node_id)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge between nodes.

        Args:
            edge: The Edge to add

        Returns:
            The added Edge

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise ValueError("Both source and target nodes must exist")

        # Use edge.id as the key in MultiDiGraph to support multiple edges
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.id,
            edge_type=edge.type,
            metadata=edge.metadata,
        )
        return edge

    def get_edges(
        self,
        source_id: UUID | None = None,
        target_id: UUID | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[Edge]:
        """Query edges with optional filters.

        Args:
            source_id: Filter to edges from this source
            target_id: Filter to edges to this target
            edge_type: Filter to edges of this type

        Returns:
            List of matching Edge objects
        """
        edges: list[Edge] = []

        if source_id and target_id:
            # MultiDiGraph returns dict of {key: data} for get_edge_data
            edge_data_dict = self._graph.get_edge_data(source_id, target_id)
            if edge_data_dict:
                for key, data in edge_data_dict.items():
                    edges.append(
                        self._edge_data_to_edge(source_id, target_id, key, data)
                    )
        elif source_id:
            # MultiDiGraph out_edges with keys=True returns (u, v, key, data)
            for _, target, key, data in self._graph.out_edges(
                source_id, data=True, keys=True
            ):
                edges.append(self._edge_data_to_edge(source_id, target, key, data))
        elif target_id:
            for source, _, key, data in self._graph.in_edges(
                target_id, data=True, keys=True
            ):
                edges.append(self._edge_data_to_edge(source, target_id, key, data))
        else:
            for source, target, key, data in self._graph.edges(data=True, keys=True):
                edges.append(self._edge_data_to_edge(source, target, key, data))

        if edge_type:
            edges = [e for e in edges if e.type == edge_type]

        return edges

    def _edge_data_to_edge(
        self, source: UUID, target: UUID, key: UUID, data: dict[str, Any]
    ) -> Edge:
        """Convert networkx edge data to Edge model."""
        return Edge(
            id=key,  # The key is the edge UUID
            source_id=source,
            target_id=target,
            type=data.get("edge_type"),
            metadata=data.get("metadata", EdgeMetadata()),
        )

    # === Query Operations ===

    def query_nodes(
        self,
        *,
        node_types: list[NodeType] | None = None,
        min_importance: float | None = None,
        max_compression_level: CompressionLevel | None = None,
        tags: set[str] | None = None,
        entity_ids: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        order_by: str = "sequence",  # "sequence", "importance", "recency"
    ) -> list[ContextNode]:
        """Query nodes with multiple filter criteria.

        Returns nodes matching ALL specified criteria.

        Args:
            node_types: Filter to these node types
            min_importance: Minimum importance score
            max_compression_level: Maximum compression level to include
            tags: Filter to nodes having any of these tags
            entity_ids: Filter to nodes referencing any of these entities
            since: Filter to nodes created at or after this time
            until: Filter to nodes created at or before this time
            limit: Maximum number of nodes to return
            order_by: Sort order - "sequence", "importance", or "recency"

        Returns:
            List of matching ContextNode objects
        """
        nodes = list(self._nodes.values())

        # Apply filters
        if node_types:
            nodes = [n for n in nodes if n.type in node_types]

        if min_importance is not None:
            nodes = [n for n in nodes if n.compute_importance() >= min_importance]

        if max_compression_level is not None:
            nodes = [n for n in nodes if n.compression_level <= max_compression_level]

        if tags:
            nodes = [n for n in nodes if tags & n.metadata.tags]

        if entity_ids:
            nodes = [
                n for n in nodes if any(e in n.metadata.entities for e in entity_ids)
            ]

        if since:
            nodes = [n for n in nodes if n.metadata.created_at >= since]

        if until:
            nodes = [n for n in nodes if n.metadata.created_at <= until]

        # Sort
        if order_by == "sequence":
            nodes.sort(key=lambda n: n.sequence_number or 0)
        elif order_by == "importance":
            nodes.sort(key=lambda n: n.compute_importance(), reverse=True)
        elif order_by == "recency":
            nodes.sort(key=lambda n: n.metadata.created_at, reverse=True)

        # Limit
        if limit:
            nodes = nodes[:limit]

        return nodes

    def get_recent(self, n: int = 10) -> list[ContextNode]:
        """Get the N most recent nodes by sequence.

        Args:
            n: Number of nodes to return

        Returns:
            List of the most recent nodes
        """
        all_nodes = self.query_nodes(order_by="sequence")
        return all_nodes[-n:] if len(all_nodes) > n else all_nodes

    def get_by_entity(self, entity_id: str) -> list[ContextNode]:
        """Get all nodes referencing a specific entity.

        Args:
            entity_id: The entity ID to search for

        Returns:
            List of nodes referencing the entity
        """
        return self.query_nodes(entity_ids=[entity_id])

    def get_neighbors(
        self,
        node_id: UUID,
        edge_types: list[EdgeType] | None = None,
        direction: str = "both",  # "in", "out", "both"
    ) -> list[ContextNode]:
        """Get neighboring nodes with optional edge type filter.

        Args:
            node_id: The UUID of the center node
            edge_types: Filter to these edge types
            direction: "in" for predecessors, "out" for successors, "both"

        Returns:
            List of neighboring nodes
        """
        neighbor_ids: set[UUID] = set()

        if direction in ("out", "both"):
            for _, target, _key, data in self._graph.out_edges(
                node_id, data=True, keys=True
            ):
                if not edge_types or data.get("edge_type") in edge_types:
                    neighbor_ids.add(target)

        if direction in ("in", "both"):
            for source, _, _key, data in self._graph.in_edges(
                node_id, data=True, keys=True
            ):
                if not edge_types or data.get("edge_type") in edge_types:
                    neighbor_ids.add(source)

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_subgraph(
        self,
        root_ids: list[UUID],
        max_depth: int = 2,
        edge_types: list[EdgeType] | None = None,
    ) -> ContextGraph:
        """Extract a subgraph starting from root nodes.

        Traverses up to max_depth edges from roots, optionally
        filtering by edge type.

        Args:
            root_ids: Starting node UUIDs for traversal
            max_depth: Maximum number of edges to traverse
            edge_types: Filter to these edge types during traversal

        Returns:
            A new ContextGraph containing the subgraph
        """
        visited: set[UUID] = set()
        to_visit: list[tuple[UUID, int]] = [
            (rid, 0) for rid in root_ids if rid in self._nodes
        ]

        while to_visit:
            node_id, depth = to_visit.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            if depth < max_depth:
                for neighbor in self.get_neighbors(node_id, edge_types):
                    if neighbor.id not in visited:
                        to_visit.append((neighbor.id, depth + 1))

        # Build new graph with extracted nodes
        subgraph = ContextGraph(session_id=f"{self.session_id}:subgraph")
        for node_id in visited:
            subgraph._nodes[node_id] = self._nodes[node_id]
            subgraph._graph.add_node(node_id)

        # Copy edges between visited nodes
        for source, target, key, data in self._graph.edges(data=True, keys=True):
            if (
                source in visited
                and target in visited
                and (not edge_types or data.get("edge_type") in edge_types)
            ):
                subgraph._graph.add_edge(source, target, key=key, **data)

        return subgraph

    # === Serialization ===

    def to_messages(
        self,
        *,
        max_tokens: int | None = None,
        tokenizer: Tokenizer | None = None,
        include_types: list[NodeType] | None = None,
        exclude_compressed: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert graph to LLM message list.

        Nodes are ordered by sequence number. If max_tokens is specified,
        truncates from the beginning (keeping recent messages).

        Args:
            max_tokens: Maximum tokens in output (truncates from start)
            tokenizer: Tokenizer for counting tokens
            include_types: Node types to include (default: message types)
            exclude_compressed: If True, exclude SUMMARIZED and EVICTED nodes

        Returns:
            List of messages in OpenAI chat completion format
        """
        nodes = self.query_nodes(
            node_types=include_types
            or [
                NodeType.MESSAGE,
                NodeType.TOOL_CALL,
                NodeType.TOOL_RESULT,
                NodeType.SYSTEM,
            ],
            max_compression_level=CompressionLevel.COMPACTED
            if exclude_compressed
            else None,
            order_by="sequence",
        )

        messages = [n.to_message_dict() for n in nodes]

        if max_tokens and tokenizer:
            # Truncate from beginning, keeping recent
            total = 0
            keep_from = 0
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = tokenizer.count_tokens(json.dumps(messages[i]))
                if total + msg_tokens > max_tokens:
                    keep_from = i + 1
                    break
                total += msg_tokens
            messages = messages[keep_from:]

        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire graph to dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "session_id": self.session_id,
            "created_at": self._created_at.isoformat(),
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [
                {
                    "id": str(key),
                    "source_id": str(s),
                    "target_id": str(t),
                    "type": d.get("edge_type").value if d.get("edge_type") else None,
                    "metadata": d.get("metadata").model_dump()
                    if d.get("metadata")
                    else {},
                }
                for s, t, key, d in self._graph.edges(data=True, keys=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextGraph:
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary representation from to_dict()

        Returns:
            A new ContextGraph instance
        """
        graph = cls(session_id=data.get("session_id"))
        graph._created_at = datetime.fromisoformat(data["created_at"])

        for node_data in data["nodes"]:
            node = ContextNode.model_validate(node_data)
            graph._nodes[node.id] = node
            graph._graph.add_node(node.id)
            graph._sequence_counter = max(
                graph._sequence_counter, (node.sequence_number or 0) + 1
            )

        for edge_data in data["edges"]:
            # Use the serialized edge id as the key for MultiDiGraph
            edge_id = UUID(edge_data["id"]) if edge_data.get("id") else uuid4()
            graph._graph.add_edge(
                UUID(edge_data["source_id"]),
                UUID(edge_data["target_id"]),
                key=edge_id,
                edge_type=EdgeType(edge_data["type"])
                if edge_data.get("type")
                else None,
                metadata=EdgeMetadata.model_validate(edge_data.get("metadata", {})),
            )

        return graph

    # === Statistics ===

    def stats(self) -> GraphStats:
        """Compute graph statistics.

        Returns:
            GraphStats with node/edge counts, token totals, and breakdowns
        """
        nodes = list(self._nodes.values())

        nodes_by_type: dict[str, int] = {}
        for nt in NodeType:
            count = sum(1 for n in nodes if n.type == nt)
            if count > 0:
                nodes_by_type[nt.value] = count

        nodes_by_compression: dict[str, int] = {}
        for cl in CompressionLevel:
            count = sum(1 for n in nodes if n.compression_level == cl)
            if count > 0:
                nodes_by_compression[cl.name] = count

        total_tokens = sum(n.token_count or 0 for n in nodes)
        avg_importance = (
            sum(n.compute_importance() for n in nodes) / len(nodes) if nodes else 0.0
        )

        return GraphStats(
            node_count=len(nodes),
            edge_count=self._graph.number_of_edges(),
            total_tokens=total_tokens,
            nodes_by_type=nodes_by_type,
            nodes_by_compression=nodes_by_compression,
            avg_importance=avg_importance,
        )

    # === Iteration ===

    def __iter__(self) -> Iterator[ContextNode]:
        """Iterate nodes in sequence order."""
        yield from sorted(self._nodes.values(), key=lambda n: n.sequence_number or 0)

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    def __contains__(self, node_id: UUID) -> bool:
        """Check if a node ID exists in the graph."""
        return node_id in self._nodes
