# Context-Core: Detailed Implementation Specification

> **Phase 1 Foundation** | Estimated: 4-6 weeks | Priority: P0
> This document provides implementable specifications for coding agents.

---

## Table of Contents
1. [Package Overview](#package-overview)
2. [Component 1: ContextGraph](#component-1-contextgraph)
3. [Component 2: EntityTracker](#component-2-entitytracker)
4. [Component 3: SemanticIndex](#component-3-semanticindex)
5. [Component 4: TokenBudget](#component-4-tokenbudget)
6. [Component 5: Tokenizer](#component-5-tokenizer)
7. [Integration Patterns](#integration-patterns)
8. [Task Breakdown](#task-breakdown)
9. [Test Specifications](#test-specifications)

---

## Package Overview

### Purpose
`context-core` provides the foundational data structures for representing, querying, and managing LLM context as a graph rather than a flat message list.

### Dependencies
```toml
[project]
name = "context-core"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "pydantic>=2.0",
    "networkx>=3.0",
    "numpy>=1.24",
]

[project.optional-dependencies]
nlp = ["spacy>=3.5", "en-core-web-sm"]
embeddings = ["sentence-transformers>=2.2"]
vector = ["chromadb>=0.4"]
tiktoken = ["tiktoken>=0.5"]
all = ["context-core[nlp,embeddings,vector,tiktoken]"]
```

### Module Structure
```
context_core/
├── __init__.py           # Public API exports
├── py.typed              # PEP 561 marker
├── graph/
│   ├── __init__.py
│   ├── types.py          # Enums: NodeType, EdgeType, CompressionLevel
│   ├── nodes.py          # ContextNode, Content, NodeMetadata
│   ├── edges.py          # Edge, EdgeMetadata
│   └── context_graph.py  # ContextGraph class
├── entities/
│   ├── __init__.py
│   ├── types.py          # EntityType enum, Entity model
│   ├── tracker.py        # EntityTracker class
│   └── backends/
│       ├── __init__.py
│       ├── base.py       # NERBackend protocol
│       ├── spacy.py      # SpacyNERBackend
│       └── pattern.py    # PatternNERBackend (regex-based)
├── semantic/
│   ├── __init__.py
│   ├── index.py          # SemanticIndex class
│   ├── embeddings.py     # EmbeddingModel protocol + implementations
│   └── stores/
│       ├── __init__.py
│       ├── base.py       # VectorStore protocol
│       ├── memory.py     # InMemoryVectorStore
│       └── chroma.py     # ChromaVectorStore
├── budget/
│   ├── __init__.py
│   ├── token_budget.py   # TokenBudget, BudgetSection
│   └── overflow.py       # OverflowStrategy enum + handlers
└── tokenizer/
    ├── __init__.py
    ├── base.py           # Tokenizer protocol
    ├── tiktoken.py       # TiktokenTokenizer
    ├── anthropic.py      # AnthropicTokenizer
    └── estimate.py       # EstimateTokenizer (fallback)
```

---

## Component 1: ContextGraph

### 1.1 Type Definitions

```python
# context_core/graph/types.py
from enum import Enum, auto
from typing import Literal

class NodeType(str, Enum):
    """Types of nodes in the context graph."""
    MESSAGE = "message"           # User or assistant message
    TOOL_CALL = "tool_call"       # Tool invocation
    TOOL_RESULT = "tool_result"   # Tool response
    ARTIFACT = "artifact"         # Code, files, structured data
    ENTITY = "entity"             # Extracted entity reference
    SUMMARY = "summary"           # Compressed representation
    SYSTEM = "system"             # System prompts, instructions
    MEMORY = "memory"             # Retrieved from long-term storage

class EdgeType(str, Enum):
    """Types of relationships between nodes."""
    TEMPORAL = "temporal"         # Sequential ordering (prev -> next)
    CAUSAL = "causal"             # A caused B
    REFERENCES = "references"     # A mentions/uses B
    SUMMARIZES = "summarizes"     # Summary node -> original nodes
    CONTRADICTS = "contradicts"   # A contradicts B
    DEPENDS_ON = "depends_on"     # A requires B
    SAME_ENTITY = "same_entity"   # Coreference link
    PARENT_CHILD = "parent_child" # Hierarchical relationship
    TOOL_IO = "tool_io"           # tool_call -> tool_result

class CompressionLevel(int, Enum):
    """Compression state of a node."""
    FULL = 0          # Original content, no compression
    COMPACTED = 1     # Reversible compression applied
    SUMMARIZED = 2    # Irreversible summarization
    EVICTED = 3       # Content externalized, pointer only

class Role(str, Enum):
    """Message roles for MESSAGE nodes."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
```

### 1.2 Node Models

```python
# context_core/graph/nodes.py
from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict

from .types import NodeType, CompressionLevel, Role

class Content(BaseModel):
    """
    Polymorphic content container for different node types.
    Uses discriminated union pattern for type safety.
    """
    model_config = ConfigDict(extra="forbid")

    # Common fields
    text: Optional[str] = None

    # MESSAGE-specific
    role: Optional[Role] = None

    # TOOL_CALL-specific
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None

    # TOOL_RESULT-specific
    tool_output: Optional[Any] = None
    is_error: bool = False

    # ARTIFACT-specific
    artifact_type: Optional[str] = None  # "code", "file", "json", etc.
    artifact_data: Optional[Any] = None
    language: Optional[str] = None       # For code artifacts
    file_path: Optional[str] = None      # For file artifacts

    # SUMMARY-specific
    summarized_node_ids: Optional[list[UUID]] = None
    summary_method: Optional[str] = None  # "hierarchical", "task_aware", etc.

    # MEMORY-specific
    memory_key: Optional[str] = None
    retrieval_score: Optional[float] = None

    # Compression tracking
    original_tokens: Optional[int] = None
    compressed_tokens: Optional[int] = None
    external_ref: Optional[str] = None  # URI for externalized content


class NodeMetadata(BaseModel):
    """
    Metadata for importance scoring, filtering, and lifecycle management.
    """
    model_config = ConfigDict(extra="allow")  # Allow custom metadata

    # Importance scoring (0.0 - 1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reference_count: int = Field(default=0, ge=0)

    # Classification
    tags: set[str] = Field(default_factory=set)
    entities: list[str] = Field(default_factory=list)  # Entity IDs

    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)

    # Preservation flags
    pinned: bool = False           # Never compress
    preserve_until: Optional[datetime] = None
    min_compression_level: CompressionLevel = CompressionLevel.FULL

    # Provenance
    source_session: Optional[str] = None
    source_agent: Optional[str] = None


class ContextNode(BaseModel):
    """
    A node in the context graph representing a unit of context.
    """
    model_config = ConfigDict(frozen=False)

    id: UUID = Field(default_factory=uuid4)
    type: NodeType
    content: Content
    metadata: NodeMetadata = Field(default_factory=NodeMetadata)

    # Compression state
    compression_level: CompressionLevel = CompressionLevel.FULL

    # Token tracking
    token_count: Optional[int] = None

    # Graph position (set by ContextGraph)
    sequence_number: Optional[int] = None

    def compute_importance(self) -> float:
        """
        Calculate composite importance score.
        Formula: 0.4 * base + 0.3 * recency + 0.2 * refs + 0.1 * type_weight
        """
        type_weights = {
            NodeType.SYSTEM: 1.0,
            NodeType.MESSAGE: 0.8,
            NodeType.TOOL_RESULT: 0.7,
            NodeType.TOOL_CALL: 0.6,
            NodeType.ARTIFACT: 0.7,
            NodeType.ENTITY: 0.5,
            NodeType.SUMMARY: 0.6,
            NodeType.MEMORY: 0.5,
        }

        base = self.metadata.importance
        recency = self.metadata.recency_score
        refs = min(self.metadata.reference_count / 10, 1.0)
        type_weight = type_weights.get(self.type, 0.5)

        return 0.4 * base + 0.3 * recency + 0.2 * refs + 0.1 * type_weight

    def to_message_dict(self) -> dict[str, Any]:
        """Convert to LLM message format (OpenAI-style)."""
        if self.type == NodeType.MESSAGE:
            return {
                "role": self.content.role.value if self.content.role else "user",
                "content": self.content.text or "",
            }
        elif self.type == NodeType.TOOL_CALL:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": str(self.id),
                    "type": "function",
                    "function": {
                        "name": self.content.tool_name,
                        "arguments": json.dumps(self.content.tool_args or {}),
                    }
                }]
            }
        elif self.type == NodeType.TOOL_RESULT:
            return {
                "role": "tool",
                "tool_call_id": str(self.id),
                "content": json.dumps(self.content.tool_output) if self.content.tool_output else "",
            }
        elif self.type == NodeType.SYSTEM:
            return {
                "role": "system",
                "content": self.content.text or "",
            }
        else:
            # For other types, render as assistant message
            return {
                "role": "assistant",
                "content": self.content.text or str(self.content.artifact_data or ""),
            }
```

### 1.3 Edge Models

```python
# context_core/graph/edges.py
from datetime import datetime
from typing import Optional, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from .types import EdgeType

class EdgeMetadata(BaseModel):
    """Metadata for graph edges."""
    weight: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    properties: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    """A directed edge between two nodes."""
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    type: EdgeType
    metadata: EdgeMetadata = Field(default_factory=EdgeMetadata)
```

### 1.4 ContextGraph Class

```python
# context_core/graph/context_graph.py
from typing import Iterator, Optional, Callable, Any
from uuid import UUID
from datetime import datetime
import json
import networkx as nx
from pydantic import BaseModel

from .types import NodeType, EdgeType, CompressionLevel
from .nodes import ContextNode, Content, NodeMetadata
from .edges import Edge, EdgeMetadata

class GraphStats(BaseModel):
    """Statistics about the context graph."""
    node_count: int
    edge_count: int
    total_tokens: int
    nodes_by_type: dict[str, int]
    nodes_by_compression: dict[str, int]
    avg_importance: float


class ContextGraph:
    """
    Graph-based context representation with typed nodes and edges.

    Thread-safe for read operations. Write operations should be
    externally synchronized in multi-threaded environments.
    """

    def __init__(self, session_id: Optional[str] = None):
        self._graph = nx.DiGraph()
        self._nodes: dict[UUID, ContextNode] = {}
        self._sequence_counter = 0
        self.session_id = session_id or str(uuid4())
        self._created_at = datetime.utcnow()

    # === CRUD Operations ===

    def add_node(
        self,
        node: ContextNode,
        *,
        connect_temporal: bool = True,
        predecessor_id: Optional[UUID] = None,
    ) -> ContextNode:
        """
        Add a node to the graph.

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
                    key=lambda n: n.sequence_number or 0
                ).id

            self.add_edge(Edge(
                source_id=pred,
                target_id=node.id,
                type=EdgeType.TEMPORAL,
            ))

        return node

    def add_message(
        self,
        role: str,
        content: str,
        **metadata_kwargs,
    ) -> ContextNode:
        """Convenience method to add a message node."""
        from .types import Role
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
        **metadata_kwargs,
    ) -> ContextNode:
        """Convenience method to add a tool call node."""
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
        **metadata_kwargs,
    ) -> ContextNode:
        """Add a tool result and link it to the tool call."""
        node = ContextNode(
            type=NodeType.TOOL_RESULT,
            content=Content(
                tool_output=output,
                is_error=is_error,
            ),
            metadata=NodeMetadata(**metadata_kwargs),
        )
        result = self.add_node(node)

        # Add TOOL_IO edge from call to result
        self.add_edge(Edge(
            source_id=tool_call_id,
            target_id=result.id,
            type=EdgeType.TOOL_IO,
        ))

        return result

    def get_node(self, node_id: UUID) -> Optional[ContextNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node(self, node_id: UUID, **updates) -> Optional[ContextNode]:
        """Update node fields. Returns updated node or None if not found."""
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

    def remove_node(self, node_id: UUID) -> Optional[ContextNode]:
        """Remove a node and its edges."""
        node = self._nodes.pop(node_id, None)
        if node and node_id in self._graph:
            self._graph.remove_node(node_id)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge between nodes."""
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise ValueError("Both source and target nodes must exist")

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            edge_id=edge.id,
            edge_type=edge.type,
            metadata=edge.metadata,
        )
        return edge

    def get_edges(
        self,
        source_id: Optional[UUID] = None,
        target_id: Optional[UUID] = None,
        edge_type: Optional[EdgeType] = None,
    ) -> list[Edge]:
        """Query edges with optional filters."""
        edges = []

        if source_id and target_id:
            edge_data = self._graph.get_edge_data(source_id, target_id)
            if edge_data:
                edges.append(self._edge_data_to_edge(source_id, target_id, edge_data))
        elif source_id:
            for _, target, data in self._graph.out_edges(source_id, data=True):
                edges.append(self._edge_data_to_edge(source_id, target, data))
        elif target_id:
            for source, _, data in self._graph.in_edges(target_id, data=True):
                edges.append(self._edge_data_to_edge(source, target_id, data))
        else:
            for source, target, data in self._graph.edges(data=True):
                edges.append(self._edge_data_to_edge(source, target, data))

        if edge_type:
            edges = [e for e in edges if e.type == edge_type]

        return edges

    def _edge_data_to_edge(self, source: UUID, target: UUID, data: dict) -> Edge:
        return Edge(
            id=data.get("edge_id"),
            source_id=source,
            target_id=target,
            type=data.get("edge_type"),
            metadata=data.get("metadata", EdgeMetadata()),
        )

    # === Query Operations ===

    def query_nodes(
        self,
        *,
        node_types: Optional[list[NodeType]] = None,
        min_importance: Optional[float] = None,
        max_compression_level: Optional[CompressionLevel] = None,
        tags: Optional[set[str]] = None,
        entity_ids: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
        order_by: str = "sequence",  # "sequence", "importance", "recency"
    ) -> list[ContextNode]:
        """
        Query nodes with multiple filter criteria.

        Returns nodes matching ALL specified criteria.
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
            nodes = [n for n in nodes if any(e in n.metadata.entities for e in entity_ids)]

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
        """Get the N most recent nodes by sequence."""
        return self.query_nodes(order_by="sequence", limit=n)[-n:]

    def get_by_entity(self, entity_id: str) -> list[ContextNode]:
        """Get all nodes referencing a specific entity."""
        return self.query_nodes(entity_ids=[entity_id])

    def get_neighbors(
        self,
        node_id: UUID,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",  # "in", "out", "both"
    ) -> list[ContextNode]:
        """Get neighboring nodes with optional edge type filter."""
        neighbor_ids = set()

        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                if not edge_types or data.get("edge_type") in edge_types:
                    neighbor_ids.add(target)

        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                if not edge_types or data.get("edge_type") in edge_types:
                    neighbor_ids.add(source)

        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def get_subgraph(
        self,
        root_ids: list[UUID],
        max_depth: int = 2,
        edge_types: Optional[list[EdgeType]] = None,
    ) -> "ContextGraph":
        """
        Extract a subgraph starting from root nodes.

        Traverses up to max_depth edges from roots, optionally
        filtering by edge type.
        """
        visited = set()
        to_visit = [(rid, 0) for rid in root_ids if rid in self._nodes]

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
        for source, target, data in self._graph.edges(data=True):
            if source in visited and target in visited:
                if not edge_types or data.get("edge_type") in edge_types:
                    subgraph._graph.add_edge(source, target, **data)

        return subgraph

    # === Serialization ===

    def to_messages(
        self,
        *,
        max_tokens: Optional[int] = None,
        tokenizer: Optional["Tokenizer"] = None,
        include_types: Optional[list[NodeType]] = None,
        exclude_compressed: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Convert graph to LLM message list.

        Nodes are ordered by sequence number. If max_tokens is specified,
        truncates from the beginning (keeping recent messages).
        """
        nodes = self.query_nodes(
            node_types=include_types or [NodeType.MESSAGE, NodeType.TOOL_CALL, NodeType.TOOL_RESULT, NodeType.SYSTEM],
            max_compression_level=CompressionLevel.COMPACTED if exclude_compressed else None,
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
        """Serialize entire graph to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at.isoformat(),
            "nodes": [n.model_dump(mode="json") for n in self._nodes.values()],
            "edges": [
                {
                    "source_id": str(s),
                    "target_id": str(t),
                    "type": d.get("edge_type").value if d.get("edge_type") else None,
                    "metadata": d.get("metadata").model_dump() if d.get("metadata") else {},
                }
                for s, t, d in self._graph.edges(data=True)
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextGraph":
        """Deserialize graph from dictionary."""
        graph = cls(session_id=data.get("session_id"))
        graph._created_at = datetime.fromisoformat(data["created_at"])

        for node_data in data["nodes"]:
            node = ContextNode.model_validate(node_data)
            graph._nodes[node.id] = node
            graph._graph.add_node(node.id)
            graph._sequence_counter = max(
                graph._sequence_counter,
                (node.sequence_number or 0) + 1
            )

        for edge_data in data["edges"]:
            graph._graph.add_edge(
                UUID(edge_data["source_id"]),
                UUID(edge_data["target_id"]),
                edge_type=EdgeType(edge_data["type"]) if edge_data.get("type") else None,
                metadata=EdgeMetadata.model_validate(edge_data.get("metadata", {})),
            )

        return graph

    # === Statistics ===

    def stats(self) -> GraphStats:
        """Compute graph statistics."""
        nodes = list(self._nodes.values())

        nodes_by_type = {}
        for nt in NodeType:
            count = sum(1 for n in nodes if n.type == nt)
            if count > 0:
                nodes_by_type[nt.value] = count

        nodes_by_compression = {}
        for cl in CompressionLevel:
            count = sum(1 for n in nodes if n.compression_level == cl)
            if count > 0:
                nodes_by_compression[cl.name] = count

        total_tokens = sum(n.token_count or 0 for n in nodes)
        avg_importance = (
            sum(n.compute_importance() for n in nodes) / len(nodes)
            if nodes else 0.0
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
        return len(self._nodes)

    def __contains__(self, node_id: UUID) -> bool:
        return node_id in self._nodes
```

---

## Component 2: EntityTracker

### 2.1 Type Definitions

```python
# context_core/entities/types.py
from enum import Enum
from typing import Optional, Any
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    """Built-in entity types."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    FILE_PATH = "file_path"
    URL = "url"
    CODE_SYMBOL = "code_symbol"      # function, class, variable names
    TECHNICAL_TERM = "technical_term"
    CUSTOM = "custom"


class Entity(BaseModel):
    """A tracked entity with metadata."""
    id: UUID = Field(default_factory=uuid4)
    type: EntityType
    canonical_name: str                    # Normalized primary name
    aliases: set[str] = Field(default_factory=set)

    # Occurrence tracking
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    mention_count: int = Field(default=1, ge=1)
    node_ids: set[UUID] = Field(default_factory=set)  # Nodes mentioning this entity

    # Importance
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # Custom properties
    properties: dict[str, Any] = Field(default_factory=dict)

    def add_mention(self, node_id: UUID, alias: Optional[str] = None):
        """Record a new mention of this entity."""
        self.mention_count += 1
        self.last_seen = datetime.utcnow()
        self.node_ids.add(node_id)
        if alias and alias != self.canonical_name:
            self.aliases.add(alias)


class EntityPattern(BaseModel):
    """Custom pattern for entity extraction."""
    name: str
    entity_type: EntityType
    pattern: str                     # Regex pattern
    importance: float = 0.5
    properties_extractor: Optional[str] = None  # Python code for extraction
```

### 2.2 NER Backend Protocol

```python
# context_core/entities/backends/base.py
from typing import Protocol, runtime_checkable
from ..types import Entity, EntityType

class EntityMention(BaseModel):
    """A single mention of an entity in text."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0


@runtime_checkable
class NERBackend(Protocol):
    """Protocol for NER implementations."""

    def extract(self, text: str) -> list[EntityMention]:
        """Extract entity mentions from text."""
        ...

    def supported_types(self) -> list[EntityType]:
        """Return entity types this backend can extract."""
        ...
```

### 2.3 EntityTracker Class

```python
# context_core/entities/tracker.py
from typing import Optional
from uuid import UUID
import re
from .types import Entity, EntityType, EntityPattern
from .backends.base import NERBackend, EntityMention

class EntityTracker:
    """
    Tracks entities across the context graph.

    Features:
    - NER-based extraction (via pluggable backends)
    - Pattern-based extraction (regex)
    - Entity resolution (merging duplicates)
    - Importance scoring based on frequency/recency
    """

    def __init__(
        self,
        ner_backend: Optional[NERBackend] = None,
        similarity_threshold: float = 0.85,
    ):
        self._entities: dict[UUID, Entity] = {}
        self._name_index: dict[str, UUID] = {}  # canonical_name -> entity_id
        self._ner_backend = ner_backend
        self._patterns: list[EntityPattern] = []
        self._similarity_threshold = similarity_threshold

        # Register default patterns
        self._register_default_patterns()

    def _register_default_patterns(self):
        """Register patterns for common technical entities."""
        self._patterns.extend([
            EntityPattern(
                name="file_path",
                entity_type=EntityType.FILE_PATH,
                pattern=r'(?:[a-zA-Z]:)?(?:[/\\][\w\-. ]+)+\.\w+',
                importance=0.7,
            ),
            EntityPattern(
                name="url",
                entity_type=EntityType.URL,
                pattern=r'https?://[^\s<>"{}|\\^`\[\]]+',
                importance=0.6,
            ),
            EntityPattern(
                name="python_import",
                entity_type=EntityType.CODE_SYMBOL,
                pattern=r'(?:from|import)\s+([\w.]+)',
                importance=0.5,
            ),
        ])

    def register_pattern(self, pattern: EntityPattern):
        """Register a custom extraction pattern."""
        self._patterns.append(pattern)

    def extract_from_text(
        self,
        text: str,
        node_id: UUID,
    ) -> list[Entity]:
        """
        Extract entities from text and register them.

        Returns list of entities found (new or existing).
        """
        mentions: list[EntityMention] = []

        # NER extraction
        if self._ner_backend:
            mentions.extend(self._ner_backend.extract(text))

        # Pattern extraction
        for pattern in self._patterns:
            for match in re.finditer(pattern.pattern, text):
                mentions.append(EntityMention(
                    text=match.group(0),
                    entity_type=pattern.entity_type,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                ))

        # Process mentions
        entities = []
        for mention in mentions:
            entity = self._resolve_or_create(mention, node_id)
            entities.append(entity)

        return entities

    def _resolve_or_create(
        self,
        mention: EntityMention,
        node_id: UUID,
    ) -> Entity:
        """Find existing entity or create new one."""
        # Normalize the name
        canonical = self._normalize_name(mention.text)

        # Check exact match
        if canonical in self._name_index:
            entity = self._entities[self._name_index[canonical]]
            entity.add_mention(node_id, mention.text)
            return entity

        # Check aliases
        for entity in self._entities.values():
            if mention.text in entity.aliases:
                entity.add_mention(node_id, mention.text)
                return entity

        # Check similarity (for fuzzy matching)
        for existing in self._entities.values():
            if existing.type == mention.entity_type:
                if self._is_similar(canonical, existing.canonical_name):
                    existing.add_mention(node_id, mention.text)
                    existing.aliases.add(canonical)
                    return existing

        # Create new entity
        entity = Entity(
            type=mention.entity_type,
            canonical_name=canonical,
            aliases={mention.text} if mention.text != canonical else set(),
            node_ids={node_id},
        )
        self._entities[entity.id] = entity
        self._name_index[canonical] = entity.id

        return entity

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for matching."""
        return name.strip().lower()

    def _is_similar(self, name1: str, name2: str) -> bool:
        """Check if two names are similar enough to be the same entity."""
        # Simple Levenshtein-based check
        if len(name1) == 0 or len(name2) == 0:
            return False

        # Quick length check
        if abs(len(name1) - len(name2)) / max(len(name1), len(name2)) > 0.3:
            return False

        # Check if one is substring of other
        if name1 in name2 or name2 in name1:
            return True

        # Could add more sophisticated similarity here
        return False

    def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def find_by_name(self, name: str) -> Optional[Entity]:
        """Find entity by name or alias."""
        canonical = self._normalize_name(name)
        if canonical in self._name_index:
            return self._entities[self._name_index[canonical]]

        for entity in self._entities.values():
            if name in entity.aliases:
                return entity

        return None

    def merge_entities(self, keep_id: UUID, merge_id: UUID) -> Optional[Entity]:
        """Merge two entities, keeping one as primary."""
        keep = self._entities.get(keep_id)
        merge = self._entities.get(merge_id)

        if not keep or not merge:
            return None

        # Merge data
        keep.aliases.update(merge.aliases)
        keep.aliases.add(merge.canonical_name)
        keep.node_ids.update(merge.node_ids)
        keep.mention_count += merge.mention_count
        keep.first_seen = min(keep.first_seen, merge.first_seen)
        keep.last_seen = max(keep.last_seen, merge.last_seen)

        # Update importance (keep higher)
        keep.importance = max(keep.importance, merge.importance)

        # Merge properties
        keep.properties.update(merge.properties)

        # Remove merged entity
        del self._entities[merge_id]
        if merge.canonical_name in self._name_index:
            del self._name_index[merge.canonical_name]

        return keep

    def get_most_important(self, n: int = 10) -> list[Entity]:
        """Get N most important entities."""
        scored = []
        for entity in self._entities.values():
            # Score: base importance + log(mentions) + recency boost
            import math
            mention_boost = min(math.log(entity.mention_count + 1) / 5, 0.3)

            hours_ago = (datetime.utcnow() - entity.last_seen).total_seconds() / 3600
            recency_boost = max(0, 0.2 - (hours_ago / 24) * 0.1)

            score = entity.importance + mention_boost + recency_boost
            scored.append((score, entity))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n]]

    def to_dict(self) -> dict:
        """Serialize tracker state."""
        return {
            "entities": [e.model_dump(mode="json") for e in self._entities.values()],
            "patterns": [p.model_dump() for p in self._patterns],
        }

    @classmethod
    def from_dict(cls, data: dict, ner_backend: Optional[NERBackend] = None) -> "EntityTracker":
        """Deserialize tracker state."""
        tracker = cls(ner_backend=ner_backend)
        tracker._patterns = [EntityPattern.model_validate(p) for p in data.get("patterns", [])]

        for entity_data in data.get("entities", []):
            entity = Entity.model_validate(entity_data)
            tracker._entities[entity.id] = entity
            tracker._name_index[entity.canonical_name] = entity.id

        return tracker
```

---

## Component 3: SemanticIndex

### 3.1 Embedding Model Protocol

```python
# context_core/semantic/embeddings.py
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        """
        Embed a batch of texts.

        Returns: Array of shape (len(texts), dimension)
        """
        ...


class SentenceTransformerEmbedding:
    """Embedding using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return self._model.encode(texts, convert_to_numpy=True)
```

### 3.2 Vector Store Protocol

```python
# context_core/semantic/stores/base.py
from typing import Protocol, Optional
from uuid import UUID
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

class SearchResult(BaseModel):
    """Result from vector search."""
    id: UUID
    score: float
    metadata: dict = {}


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends."""

    def add(
        self,
        ids: list[UUID],
        embeddings: NDArray[np.float32],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """Add vectors to the store."""
        ...

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        ...

    def delete(self, ids: list[UUID]) -> None:
        """Delete vectors by ID."""
        ...

    def get(self, ids: list[UUID]) -> NDArray[np.float32]:
        """Get vectors by ID."""
        ...
```

### 3.3 InMemoryVectorStore

```python
# context_core/semantic/stores/memory.py
from typing import Optional
from uuid import UUID
import numpy as np
from numpy.typing import NDArray

from .base import VectorStore, SearchResult

class InMemoryVectorStore:
    """Simple in-memory vector store using numpy."""

    def __init__(self, dimension: int):
        self._dimension = dimension
        self._vectors: dict[UUID, NDArray[np.float32]] = {}
        self._metadata: dict[UUID, dict] = {}

    def add(
        self,
        ids: list[UUID],
        embeddings: NDArray[np.float32],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        metadata = metadata or [{} for _ in ids]
        for i, id_ in enumerate(ids):
            self._vectors[id_] = embeddings[i]
            self._metadata[id_] = metadata[i]

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        if not self._vectors:
            return []

        # Stack all vectors
        ids = list(self._vectors.keys())
        vectors = np.stack([self._vectors[id_] for id_ in ids])

        # Compute cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        similarities = vectors_norm @ query_norm

        # Apply filter if provided
        if filter:
            mask = np.ones(len(ids), dtype=bool)
            for key, value in filter.items():
                for i, id_ in enumerate(ids):
                    if self._metadata[id_].get(key) != value:
                        mask[i] = False
            similarities = np.where(mask, similarities, -1)

        # Get top k
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [
            SearchResult(
                id=ids[i],
                score=float(similarities[i]),
                metadata=self._metadata[ids[i]],
            )
            for i in top_indices
            if similarities[i] > 0
        ]

    def delete(self, ids: list[UUID]) -> None:
        for id_ in ids:
            self._vectors.pop(id_, None)
            self._metadata.pop(id_, None)

    def get(self, ids: list[UUID]) -> NDArray[np.float32]:
        return np.stack([self._vectors[id_] for id_ in ids if id_ in self._vectors])
```

### 3.4 SemanticIndex Class

```python
# context_core/semantic/index.py
from typing import Optional
from uuid import UUID
import numpy as np
from numpy.typing import NDArray

from .embeddings import EmbeddingModel
from .stores.base import VectorStore, SearchResult
from .stores.memory import InMemoryVectorStore
from ..graph.nodes import ContextNode

class SemanticIndex:
    """
    Semantic indexing for context nodes.

    Features:
    - Embed and index node content
    - Semantic similarity search
    - Duplicate detection
    - Clustering
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: Optional[VectorStore] = None,
    ):
        self._embedding_model = embedding_model
        self._vector_store = vector_store or InMemoryVectorStore(
            dimension=embedding_model.dimension
        )
        self._node_texts: dict[UUID, str] = {}  # Cache of node text content

    def index_node(self, node: ContextNode) -> None:
        """Index a single node."""
        text = self._extract_text(node)
        if not text:
            return

        embedding = self._embedding_model.embed([text])[0]
        self._vector_store.add(
            ids=[node.id],
            embeddings=np.expand_dims(embedding, 0),
            metadata=[{
                "type": node.type.value,
                "compression_level": node.compression_level.value,
            }],
        )
        self._node_texts[node.id] = text

    def index_nodes(self, nodes: list[ContextNode]) -> None:
        """Batch index multiple nodes."""
        texts = []
        valid_nodes = []

        for node in nodes:
            text = self._extract_text(node)
            if text:
                texts.append(text)
                valid_nodes.append(node)

        if not texts:
            return

        embeddings = self._embedding_model.embed(texts)
        self._vector_store.add(
            ids=[n.id for n in valid_nodes],
            embeddings=embeddings,
            metadata=[{
                "type": n.type.value,
                "compression_level": n.compression_level.value,
            } for n in valid_nodes],
        )

        for node, text in zip(valid_nodes, texts):
            self._node_texts[node.id] = text

    def _extract_text(self, node: ContextNode) -> Optional[str]:
        """Extract indexable text from a node."""
        content = node.content

        if content.text:
            return content.text
        elif content.tool_name:
            return f"{content.tool_name}: {content.tool_args}"
        elif content.tool_output:
            return str(content.tool_output)[:1000]  # Limit length
        elif content.artifact_data:
            return str(content.artifact_data)[:1000]

        return None

    def search(
        self,
        query: str,
        k: int = 10,
        min_score: float = 0.0,
        filter: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Search for semantically similar nodes.

        Args:
            query: Search query text
            k: Maximum results to return
            min_score: Minimum similarity score (0-1)
            filter: Metadata filter (e.g., {"type": "message"})

        Returns:
            List of SearchResult with node IDs and scores
        """
        query_embedding = self._embedding_model.embed([query])[0]
        results = self._vector_store.search(query_embedding, k=k, filter=filter)
        return [r for r in results if r.score >= min_score]

    def find_duplicates(
        self,
        threshold: float = 0.95,
    ) -> list[tuple[UUID, UUID, float]]:
        """
        Find near-duplicate nodes.

        Returns list of (node_id_1, node_id_2, similarity) tuples.
        """
        duplicates = []
        node_ids = list(self._node_texts.keys())

        # Get all embeddings
        embeddings = self._vector_store.get(node_ids)

        # Compute pairwise similarity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = embeddings_norm @ embeddings_norm.T

        # Find pairs above threshold
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                if similarity_matrix[i, j] >= threshold:
                    duplicates.append((
                        node_ids[i],
                        node_ids[j],
                        float(similarity_matrix[i, j]),
                    ))

        return sorted(duplicates, key=lambda x: x[2], reverse=True)

    def remove_node(self, node_id: UUID) -> None:
        """Remove a node from the index."""
        self._vector_store.delete([node_id])
        self._node_texts.pop(node_id, None)
```

---

## Component 4: TokenBudget

```python
# context_core/budget/token_budget.py
from typing import Optional, Callable
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class BudgetSection(BaseModel):
    """A named section of the token budget."""
    name: str
    allocated: int          # Maximum tokens for this section
    used: int = 0           # Currently used tokens
    priority: int = 0       # Higher = compress last

    @property
    def available(self) -> int:
        return max(0, self.allocated - self.used)

    @property
    def usage_ratio(self) -> float:
        return self.used / self.allocated if self.allocated > 0 else 0.0


class BudgetAlert(BaseModel):
    """An alert triggered by budget conditions."""
    level: str              # "warning", "critical"
    section: Optional[str]  # None for global alerts
    message: str
    usage_ratio: float
    triggered_at: datetime = Field(default_factory=datetime.utcnow)


class OverflowStrategy(str, Enum):
    """Strategy for handling budget overflow."""
    ERROR = "error"                 # Raise exception
    TRUNCATE_OLD = "truncate_old"   # Remove oldest content
    COMPRESS = "compress"           # Trigger compression
    OVERFLOW = "overflow"           # Allow temporary overflow


class TokenBudget:
    """
    Manages token budget allocation across sections.

    Features:
    - Section-based allocation (system, conversation, tools, etc.)
    - Pre-rot threshold monitoring
    - Overflow handling strategies
    - Alert callbacks
    """

    DEFAULT_SECTIONS = {
        "system": {"allocation_ratio": 0.10, "priority": 100},
        "conversation": {"allocation_ratio": 0.60, "priority": 50},
        "tools": {"allocation_ratio": 0.20, "priority": 30},
        "memory": {"allocation_ratio": 0.10, "priority": 20},
    }

    def __init__(
        self,
        total_tokens: int,
        pre_rot_warning: float = 0.50,    # Warn at 50% usage
        pre_rot_trigger: float = 0.65,    # Trigger compression at 65%
        overflow_strategy: OverflowStrategy = OverflowStrategy.COMPRESS,
        on_alert: Optional[Callable[[BudgetAlert], None]] = None,
    ):
        self.total_tokens = total_tokens
        self.pre_rot_warning = pre_rot_warning
        self.pre_rot_trigger = pre_rot_trigger
        self.overflow_strategy = overflow_strategy
        self._on_alert = on_alert

        self._sections: dict[str, BudgetSection] = {}
        self._alerts: list[BudgetAlert] = []
        self._warning_fired = False
        self._trigger_fired = False

        # Initialize default sections
        self._init_default_sections()

    def _init_default_sections(self):
        """Initialize default budget sections."""
        for name, config in self.DEFAULT_SECTIONS.items():
            self.add_section(
                name=name,
                allocated=int(self.total_tokens * config["allocation_ratio"]),
                priority=config["priority"],
            )

    def add_section(
        self,
        name: str,
        allocated: int,
        priority: int = 0,
    ) -> BudgetSection:
        """Add a budget section."""
        section = BudgetSection(
            name=name,
            allocated=allocated,
            priority=priority,
        )
        self._sections[name] = section
        return section

    def allocate(
        self,
        section: str,
        tokens: int,
        force: bool = False,
    ) -> bool:
        """
        Allocate tokens to a section.

        Args:
            section: Section name
            tokens: Number of tokens to allocate
            force: If True, allow overflow

        Returns:
            True if allocation succeeded, False if would overflow
        """
        if section not in self._sections:
            raise ValueError(f"Unknown section: {section}")

        sec = self._sections[section]
        new_used = sec.used + tokens

        if new_used > sec.allocated and not force:
            if self.overflow_strategy == OverflowStrategy.ERROR:
                raise ValueError(f"Section {section} would overflow")
            elif self.overflow_strategy == OverflowStrategy.OVERFLOW:
                pass  # Allow it
            else:
                return False

        sec.used = new_used
        self._check_thresholds()
        return True

    def deallocate(self, section: str, tokens: int) -> None:
        """Release tokens from a section."""
        if section not in self._sections:
            raise ValueError(f"Unknown section: {section}")

        sec = self._sections[section]
        sec.used = max(0, sec.used - tokens)

        # Reset trigger states if we've dropped below thresholds
        if self.usage_ratio < self.pre_rot_warning:
            self._warning_fired = False
        if self.usage_ratio < self.pre_rot_trigger:
            self._trigger_fired = False

    def _check_thresholds(self):
        """Check and fire alerts for threshold crossings."""
        ratio = self.usage_ratio

        if ratio >= self.pre_rot_trigger and not self._trigger_fired:
            self._trigger_fired = True
            alert = BudgetAlert(
                level="critical",
                section=None,
                message=f"Pre-rot trigger reached: {ratio:.1%} usage",
                usage_ratio=ratio,
            )
            self._alerts.append(alert)
            if self._on_alert:
                self._on_alert(alert)

        elif ratio >= self.pre_rot_warning and not self._warning_fired:
            self._warning_fired = True
            alert = BudgetAlert(
                level="warning",
                section=None,
                message=f"Pre-rot warning: {ratio:.1%} usage",
                usage_ratio=ratio,
            )
            self._alerts.append(alert)
            if self._on_alert:
                self._on_alert(alert)

    @property
    def used_tokens(self) -> int:
        """Total tokens used across all sections."""
        return sum(s.used for s in self._sections.values())

    @property
    def available_tokens(self) -> int:
        """Total tokens available."""
        return max(0, self.total_tokens - self.used_tokens)

    @property
    def usage_ratio(self) -> float:
        """Overall usage ratio (0.0 - 1.0+)."""
        return self.used_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

    @property
    def is_pre_rot_warning(self) -> bool:
        """True if usage has crossed warning threshold."""
        return self.usage_ratio >= self.pre_rot_warning

    @property
    def is_pre_rot_trigger(self) -> bool:
        """True if usage has crossed trigger threshold."""
        return self.usage_ratio >= self.pre_rot_trigger

    def get_section(self, name: str) -> Optional[BudgetSection]:
        """Get a section by name."""
        return self._sections.get(name)

    def get_compression_priority(self) -> list[str]:
        """
        Get sections in compression priority order.

        Lower priority sections should be compressed first.
        """
        return [
            s.name for s in
            sorted(self._sections.values(), key=lambda s: s.priority)
        ]

    def to_dict(self) -> dict:
        """Serialize budget state."""
        return {
            "total_tokens": self.total_tokens,
            "pre_rot_warning": self.pre_rot_warning,
            "pre_rot_trigger": self.pre_rot_trigger,
            "overflow_strategy": self.overflow_strategy.value,
            "sections": {
                name: sec.model_dump()
                for name, sec in self._sections.items()
            },
            "alerts": [a.model_dump(mode="json") for a in self._alerts],
        }

    def summary(self) -> str:
        """Human-readable budget summary."""
        lines = [
            f"Token Budget: {self.used_tokens:,} / {self.total_tokens:,} ({self.usage_ratio:.1%})",
            f"Status: {'CRITICAL' if self.is_pre_rot_trigger else 'WARNING' if self.is_pre_rot_warning else 'OK'}",
            "",
            "Sections:",
        ]
        for sec in sorted(self._sections.values(), key=lambda s: -s.priority):
            status = "OVER" if sec.used > sec.allocated else "OK"
            lines.append(
                f"  {sec.name}: {sec.used:,} / {sec.allocated:,} ({sec.usage_ratio:.1%}) [{status}]"
            )

        return "\n".join(lines)
```

---

## Component 5: Tokenizer

```python
# context_core/tokenizer/base.py
from typing import Protocol, Optional, runtime_checkable
from functools import lru_cache

@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizer implementations."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        ...


class TiktokenTokenizer:
    """Tokenizer using tiktoken (OpenAI models)."""

    def __init__(self, model: str = "gpt-4"):
        import tiktoken
        self._encoding = tiktoken.encoding_for_model(model)

    @lru_cache(maxsize=10000)
    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])


class EstimateTokenizer:
    """Fallback tokenizer using character estimation."""

    def __init__(self, chars_per_token: float = 4.0):
        self._chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        return int(len(text) / self._chars_per_token)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        max_chars = int(max_tokens * self._chars_per_token)
        return text[:max_chars]


def get_tokenizer(model: str) -> Tokenizer:
    """Factory function to get appropriate tokenizer for model."""
    if "gpt" in model.lower() or "text-" in model.lower():
        try:
            return TiktokenTokenizer(model)
        except Exception:
            pass

    # Fallback to estimate
    return EstimateTokenizer()
```

---

## Integration Patterns

### Creating a ContextGraph with Full Stack

```python
from context_core import (
    ContextGraph, EntityTracker, SemanticIndex, TokenBudget,
    get_tokenizer, SentenceTransformerEmbedding,
)

# Initialize components
tokenizer = get_tokenizer("gpt-4")
embedding_model = SentenceTransformerEmbedding()
entity_tracker = EntityTracker()
semantic_index = SemanticIndex(embedding_model)
budget = TokenBudget(
    total_tokens=128_000,
    pre_rot_warning=0.50,
    pre_rot_trigger=0.65,
    on_alert=lambda alert: print(f"ALERT: {alert.message}"),
)

# Create graph
graph = ContextGraph()

# Add a message and process it
node = graph.add_message(role="user", content="Please analyze the file /src/main.py")

# Extract entities
entities = entity_tracker.extract_from_text(node.content.text, node.id)
node.metadata.entities = [str(e.id) for e in entities]

# Index semantically
semantic_index.index_node(node)

# Track tokens
node.token_count = tokenizer.count_tokens(node.content.text)
budget.allocate("conversation", node.token_count)

# Check status
print(budget.summary())
```

---

## Task Breakdown

### Week 1-2: ContextGraph Foundation

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| CG-001 | Create `graph/types.py` with enums | 2 | None | All enums defined, tests pass |
| CG-002 | Implement `Content` model | 4 | CG-001 | Pydantic validation works, all fields |
| CG-003 | Implement `NodeMetadata` model | 3 | CG-001 | Importance scoring works |
| CG-004 | Implement `ContextNode` model | 4 | CG-002, CG-003 | `to_message_dict()` works for all types |
| CG-005 | Implement `Edge` and `EdgeMetadata` | 2 | CG-001 | Edge creation works |
| CG-006 | Implement `ContextGraph` CRUD | 8 | CG-004, CG-005 | add/get/update/remove work |
| CG-007 | Implement `ContextGraph` queries | 6 | CG-006 | All query filters work |
| CG-008 | Implement `ContextGraph` subgraph | 4 | CG-006 | Subgraph extraction works |
| CG-009 | Implement serialization | 4 | CG-006 | `to_dict`/`from_dict` round-trip |
| CG-010 | Implement `to_messages` | 4 | CG-006 | LLM-ready message list |
| CG-011 | Write unit tests | 8 | CG-001 to CG-010 | 90%+ coverage |
| CG-012 | Write integration tests | 4 | CG-011 | End-to-end scenarios pass |

### Week 3: EntityTracker

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| ET-001 | Create `entities/types.py` | 3 | None | Entity, EntityType, EntityPattern |
| ET-002 | Define `NERBackend` protocol | 2 | ET-001 | Protocol with extract method |
| ET-003 | Implement `PatternNERBackend` | 4 | ET-002 | Regex extraction works |
| ET-004 | Implement `SpacyNERBackend` | 4 | ET-002 | spaCy integration works |
| ET-005 | Implement `EntityTracker` core | 6 | ET-001 to ET-004 | Extract and track entities |
| ET-006 | Implement entity resolution | 4 | ET-005 | Merge duplicates, fuzzy match |
| ET-007 | Implement importance scoring | 2 | ET-005 | Score calculation works |
| ET-008 | Write unit tests | 6 | ET-001 to ET-007 | 90%+ coverage |

### Week 4: SemanticIndex

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| SI-001 | Define `EmbeddingModel` protocol | 2 | None | Protocol defined |
| SI-002 | Implement `SentenceTransformerEmbedding` | 3 | SI-001 | Embedding works |
| SI-003 | Define `VectorStore` protocol | 2 | None | Protocol defined |
| SI-004 | Implement `InMemoryVectorStore` | 4 | SI-003 | Add/search/delete work |
| SI-005 | Implement `ChromaVectorStore` | 4 | SI-003 | ChromaDB integration |
| SI-006 | Implement `SemanticIndex` | 6 | SI-001 to SI-005 | Index and search nodes |
| SI-007 | Implement duplicate detection | 3 | SI-006 | Find duplicates works |
| SI-008 | Write unit tests | 6 | SI-001 to SI-007 | 90%+ coverage |

### Week 5: TokenBudget

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| TB-001 | Implement `BudgetSection` | 2 | None | Section model works |
| TB-002 | Implement `TokenBudget` core | 4 | TB-001 | Allocation/deallocation |
| TB-003 | Implement pre-rot detection | 3 | TB-002 | Thresholds fire alerts |
| TB-004 | Implement overflow strategies | 3 | TB-002 | All strategies work |
| TB-005 | Implement `Tokenizer` protocol | 2 | None | Protocol defined |
| TB-006 | Implement `TiktokenTokenizer` | 3 | TB-005 | Token counting works |
| TB-007 | Implement `EstimateTokenizer` | 2 | TB-005 | Fallback works |
| TB-008 | Write unit tests | 6 | TB-001 to TB-007 | 90%+ coverage |

### Week 6: Integration & Polish

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| INT-001 | Create public API (`__init__.py`) | 3 | All above | Clean exports |
| INT-002 | Write integration tests | 8 | INT-001 | Full stack works together |
| INT-003 | Performance benchmarks | 4 | INT-001 | Baseline metrics |
| INT-004 | Documentation | 6 | INT-001 | API docs, examples |
| INT-005 | Package setup (`pyproject.toml`) | 2 | INT-001 | pip install works |
| INT-006 | CI/CD setup | 4 | INT-005 | GitHub Actions |

---

## Test Specifications

### Unit Test Requirements

Each component must have:
- **Happy path tests**: Normal usage scenarios
- **Edge cases**: Empty inputs, boundary conditions
- **Error handling**: Invalid inputs, missing data
- **Serialization round-trips**: `to_dict` -> `from_dict` equality

### Example Test Cases

```python
# tests/test_context_graph.py
import pytest
from context_core.graph import ContextGraph, ContextNode, NodeType, EdgeType, Content

class TestContextGraph:
    def test_add_message_creates_node(self):
        graph = ContextGraph()
        node = graph.add_message(role="user", content="Hello")

        assert node.id in graph
        assert node.type == NodeType.MESSAGE
        assert node.content.text == "Hello"
        assert node.sequence_number == 0

    def test_temporal_edges_auto_created(self):
        graph = ContextGraph()
        n1 = graph.add_message(role="user", content="First")
        n2 = graph.add_message(role="assistant", content="Second")

        edges = graph.get_edges(source_id=n1.id, edge_type=EdgeType.TEMPORAL)
        assert len(edges) == 1
        assert edges[0].target_id == n2.id

    def test_query_by_type(self):
        graph = ContextGraph()
        graph.add_message(role="user", content="User msg")
        graph.add_message(role="assistant", content="Assistant msg")
        graph.add_tool_call("read_file", {"path": "/test"})

        messages = graph.query_nodes(node_types=[NodeType.MESSAGE])
        assert len(messages) == 2

        tools = graph.query_nodes(node_types=[NodeType.TOOL_CALL])
        assert len(tools) == 1

    def test_serialization_roundtrip(self):
        graph = ContextGraph()
        graph.add_message(role="user", content="Test")
        graph.add_message(role="assistant", content="Response")

        data = graph.to_dict()
        restored = ContextGraph.from_dict(data)

        assert len(restored) == len(graph)
        assert restored.session_id == graph.session_id

    def test_to_messages_format(self):
        graph = ContextGraph()
        graph.add_message(role="user", content="Hello")
        graph.add_message(role="assistant", content="Hi there")

        messages = graph.to_messages()

        assert messages == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]


class TestTokenBudget:
    def test_pre_rot_warning_fires(self):
        alerts = []
        budget = TokenBudget(
            total_tokens=1000,
            pre_rot_warning=0.50,
            on_alert=lambda a: alerts.append(a),
        )

        # Allocate 51% to trigger warning
        budget.allocate("conversation", 510, force=True)

        assert len(alerts) == 1
        assert alerts[0].level == "warning"
        assert budget.is_pre_rot_warning

    def test_section_priority_order(self):
        budget = TokenBudget(total_tokens=1000)

        priority_order = budget.get_compression_priority()

        # Memory (20) < Tools (30) < Conversation (50) < System (100)
        assert priority_order[0] == "memory"
        assert priority_order[-1] == "system"
```

---

## Validation Criteria

### Definition of Done for Phase 1

1. **All models implemented** with Pydantic validation
2. **All protocols defined** with at least one implementation
3. **Unit test coverage** >= 90%
4. **Integration tests** pass with full stack
5. **Documentation** complete for public API
6. **Performance benchmarks** establish baseline:
   - ContextGraph: 10,000 nodes in < 1s
   - SemanticIndex: 1,000 searches in < 10s
   - TokenBudget: 100,000 allocations in < 1s
7. **Package installable** via pip

---

*This specification is designed to be directly implementable by coding agents. Each section contains complete, copy-paste-ready code with type hints and docstrings.*
