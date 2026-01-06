# ContextEngine: API Contracts Between Packages

> This document defines the interfaces and contracts between packages to ensure loose coupling and clear dependencies.

---

## Package Dependency Graph

```
                    ┌─────────────────────────────────┐
                    │         context-engine          │
                    │         (Unified SDK)           │
                    └───────────────┬─────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│context-multi- │         │ context-tools   │         │ integrations    │
│    agent      │         │                 │         │ (langchain,etc) │
└───────┬───────┘         └────────┬────────┘         └────────┬────────┘
        │                          │                           │
        └──────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   context-memory    │
                        └──────────┬──────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌───────────────┐         ┌───────────────────┐      ┌────────────────┐
│context-       │         │   context-core    │      │context-observe │
│compression    │         │                   │      │                │
└───────┬───────┘         └─────────────┬─────┘      └────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
                ┌───────────────────┐
                │    pydantic       │
                │    networkx       │
                │    numpy          │
                └───────────────────┘
```

---

## Core Interfaces

### 1. context-core Exports

The foundational package exports these types for use by other packages:

```python
# context_core/__init__.py

# Graph components
from .graph import (
    ContextGraph,
    ContextNode,
    Content,
    NodeMetadata,
    Edge,
    EdgeMetadata,
    NodeType,
    EdgeType,
    CompressionLevel,
    Role,
    GraphStats,
)

# Entity tracking
from .entities import (
    EntityTracker,
    Entity,
    EntityType,
    EntityPattern,
    NERBackend,
)

# Semantic indexing
from .semantic import (
    SemanticIndex,
    EmbeddingModel,
    VectorStore,
    SearchResult,
)

# Token management
from .budget import (
    TokenBudget,
    BudgetSection,
    BudgetAlert,
    OverflowStrategy,
)

# Tokenization
from .tokenizer import (
    Tokenizer,
    get_tokenizer,
)
```

### 2. context-core → context-compression Contract

**context-compression expects:**

```python
from context_core.graph import (
    ContextGraph,      # Must support: __iter__, get_node, remove_node, add_node
    ContextNode,       # Must have: id, type, content, metadata, compression_level, token_count
    NodeType,          # Enum with: MESSAGE, TOOL_CALL, TOOL_RESULT, ARTIFACT, etc.
    EdgeType,          # Enum with: TEMPORAL, CAUSAL, REFERENCES, SUMMARIZES
    CompressionLevel,  # Enum with: FULL, COMPACTED, SUMMARIZED, EVICTED
)

from context_core.semantic import (
    SemanticIndex,     # Must support: find_duplicates(threshold) -> list[tuple[UUID, UUID, float]]
)

from context_core.entities import (
    EntityTracker,     # Must support: get_most_important(n) -> list[Entity]
)
```

**Interface contract for ContextGraph used by compression:**

```python
class ContextGraphProtocol(Protocol):
    """What context-compression needs from ContextGraph."""

    session_id: str

    def __iter__(self) -> Iterator[ContextNode]: ...
    def __len__(self) -> int: ...
    def __contains__(self, node_id: UUID) -> bool: ...

    def get_node(self, node_id: UUID) -> Optional[ContextNode]: ...
    def add_node(self, node: ContextNode, connect_temporal: bool = True) -> ContextNode: ...
    def remove_node(self, node_id: UUID) -> Optional[ContextNode]: ...
    def add_edge(self, edge: Edge) -> Edge: ...

    def query_nodes(
        self,
        node_types: Optional[list[NodeType]] = None,
        max_compression_level: Optional[CompressionLevel] = None,
        order_by: str = "sequence",
    ) -> list[ContextNode]: ...

    def get_recent(self, n: int) -> list[ContextNode]: ...
    def to_dict(self) -> dict: ...
```

### 3. context-core → context-memory Contract

**context-memory expects:**

```python
from context_core.graph import (
    ContextGraph,      # Must support: to_dict(), from_dict()
    ContextNode,       # Must be Pydantic model with model_dump() and model_validate()
    NodeType,          # For metadata categorization
)

from context_core.semantic import (
    SemanticIndex,     # For semantic retrieval: search(query, k, min_score)
)

from context_core.entities import (
    EntityTracker,     # For entity retrieval: get_entity(), extract_from_text()
)
```

**Interface contract for ContextNode serialization:**

```python
class ContextNodeProtocol(Protocol):
    """What context-memory needs from ContextNode."""

    id: UUID
    type: NodeType
    token_count: Optional[int]
    metadata: NodeMetadata

    def model_dump(self, mode: str = "python") -> dict: ...

    @classmethod
    def model_validate(cls, data: dict) -> "ContextNode": ...
```

### 4. context-compression → context-memory Contract

**context-memory expects from compression (for externalization):**

```python
from context_compression.strategies.lossless import ExternalStorage

# MemoryStore backends must implement ExternalStorage for integration
class MemoryStoreAsExternalStorage:
    """Adapter to use MemoryStore as ExternalStorage."""

    def __init__(self, memory_store: MemoryStore, session_id: str):
        self._store = memory_store
        self._session_id = session_id

    def store(self, key: str, data: bytes) -> str:
        # Store and return URI
        ...

    def retrieve(self, uri: str) -> bytes:
        # Retrieve by URI
        ...
```

### 5. context-tools → context-core Contract

**context-tools expects:**

```python
from context_core.semantic import (
    SemanticIndex,     # For semantic cache matching
    EmbeddingModel,    # For generating cache key embeddings
)

from context_core.tokenizer import (
    Tokenizer,         # For token counting in cache
)
```

### 6. context-multiagent → context-core Contract

**context-multiagent expects:**

```python
from context_core.graph import (
    ContextGraph,      # Full graph with: to_dict(), from_dict(), get_subgraph()
    ContextNode,       # For briefing generation
    NodeType,          # For filtering shared context
)
```

### 7. context-observe → All Packages Contract

**All packages should accept optional tracer/metrics:**

```python
from context_observe import (
    ContextTracer,     # Optional: for span creation
    ContextMetrics,    # Optional: for metrics recording
    ContextEvents,     # Optional: for structured logging
)

# Example: context-compression with observability
class CompressionPipeline:
    def __init__(
        self,
        strategies: list[CompressionStrategy],
        tracer: Optional[ContextTracer] = None,
        metrics: Optional[ContextMetrics] = None,
        events: Optional[ContextEvents] = None,
    ):
        self._tracer = tracer
        self._metrics = metrics
        self._events = events

    def compress(self, graph: ContextGraph, ...) -> list[CompressionResult]:
        if self._tracer:
            with self._tracer.trace_compression("pipeline", graph.session_id, ...):
                result = self._do_compress(graph, ...)
        else:
            result = self._do_compress(graph, ...)

        if self._metrics:
            self._metrics.record_compression(...)

        if self._events:
            self._events.compression_completed(...)

        return result
```

---

## Protocol Definitions

### Storage Protocols

```python
# Defined in context_memory/store.py
@runtime_checkable
class MemoryStore(Protocol):
    """Universal storage protocol."""

    async def store(self, node: ContextNode, session_id: str, metadata: Optional[StorageMetadata] = None) -> StorageKey: ...
    async def retrieve(self, key: StorageKey) -> Optional[ContextNode]: ...
    async def delete(self, key: StorageKey) -> bool: ...
    async def exists(self, key: StorageKey) -> bool: ...
    async def get_metadata(self, key: StorageKey) -> Optional[StorageMetadata]: ...
    async def list_keys(self, session_id: str, **filters) -> list[StorageKey]: ...
    async def stats(self, session_id: Optional[str] = None) -> StorageStats: ...
    async def close(self) -> None: ...
```

### Retrieval Protocols

```python
# Defined in context_memory/retrieval/base.py
@runtime_checkable
class RetrievalStrategy(Protocol):
    """Retrieval strategy protocol."""

    @property
    def name(self) -> str: ...

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Optional[ContextGraph] = None,
    ) -> list[RetrievalResult]: ...
```

### Compression Protocols

```python
# Defined in context_compression/strategies/base.py
@runtime_checkable
class CompressionStrategy(Protocol):
    """Compression strategy protocol."""

    @property
    def name(self) -> str: ...

    @property
    def tier(self) -> CompressionTier: ...

    @property
    def priority(self) -> int: ...

    def estimate_savings(self, graph: ContextGraph, target_node_ids: Optional[list[UUID]] = None) -> int: ...

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult: ...

    def can_apply(self, graph: ContextGraph) -> bool: ...
```

### Embedding Protocols

```python
# Defined in context_core/semantic/embeddings.py
@runtime_checkable
class EmbeddingModel(Protocol):
    """Embedding model protocol."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...
```

### NER Protocols

```python
# Defined in context_core/entities/backends/base.py
@runtime_checkable
class NERBackend(Protocol):
    """NER backend protocol."""

    def extract(self, text: str) -> list[EntityMention]: ...
    def supported_types(self) -> list[EntityType]: ...
```

### Transport Protocols

```python
# Defined in context_multiagent/transport/base.py
@runtime_checkable
class MessageTransport(Protocol):
    """Message transport protocol for multi-agent."""

    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    async def publish(self, topic: str, message: dict) -> None: ...
    async def subscribe(self, topics: Set[str]) -> None: ...
    async def unsubscribe(self, topics: Set[str]) -> None: ...
    async def subscribe_all(self) -> AsyncIterator[dict]: ...
```

---

## Data Transfer Objects

### Shared DTOs

These models are shared across packages:

```python
# Re-exported by context-core for all packages
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class NodeReference(BaseModel):
    """Lightweight reference to a node."""
    id: UUID
    type: str
    session_id: str


class TokenCount(BaseModel):
    """Token count with breakdown."""
    total: int
    by_type: dict[str, int]
    by_section: dict[str, int]


class OperationResult(BaseModel):
    """Generic operation result."""
    success: bool
    message: Optional[str] = None
    data: Optional[dict] = None
    duration_ms: float = 0.0
```

---

## Error Handling Contract

All packages should use these exception types:

```python
# context_core/exceptions.py (exported by all packages)

class ContextEngineError(Exception):
    """Base exception for all context-engine errors."""
    pass


class NodeNotFoundError(ContextEngineError):
    """Node not found in graph or storage."""
    def __init__(self, node_id: UUID):
        self.node_id = node_id
        super().__init__(f"Node not found: {node_id}")


class StorageError(ContextEngineError):
    """Storage operation failed."""
    pass


class CompressionError(ContextEngineError):
    """Compression operation failed."""
    pass


class TokenBudgetExceededError(ContextEngineError):
    """Token budget would be exceeded."""
    def __init__(self, required: int, available: int):
        self.required = required
        self.available = available
        super().__init__(f"Token budget exceeded: need {required}, have {available}")


class RecoveryError(ContextEngineError):
    """Context recovery failed."""
    pass


class HandoffError(ContextEngineError):
    """Agent handoff failed."""
    pass
```

---

## Versioning Contract

All packages follow semantic versioning with these compatibility rules:

1. **Major version (X.0.0)**: Breaking changes to protocols or public API
2. **Minor version (0.X.0)**: New features, backward compatible
3. **Patch version (0.0.X)**: Bug fixes only

**Cross-package compatibility:**
- All packages in the same major version are compatible
- context-core is the version anchor
- Other packages depend on context-core with `>=X.0.0,<Y.0.0`

---

## Testing Contracts

### Mock Implementations

Each package provides test doubles:

```python
# context_core/testing.py
class MockContextGraph:
    """In-memory ContextGraph for testing."""
    ...

class MockSemanticIndex:
    """SemanticIndex with deterministic embeddings."""
    ...

class MockEntityTracker:
    """EntityTracker with pattern-only extraction."""
    ...


# context_memory/testing.py
class MockMemoryStore:
    """In-memory MemoryStore for testing."""
    ...


# context_compression/testing.py
class MockLLMSummarizer:
    """Summarizer that returns truncated input."""
    ...


# context_multiagent/testing.py
class MockMessageTransport:
    """In-memory transport for testing."""
    ...
```

### Integration Test Fixtures

```python
# conftest.py (shared across all packages)

@pytest.fixture
def context_graph():
    """Create a populated test graph."""
    graph = ContextGraph(session_id="test-session")
    graph.add_message(role="user", content="Hello")
    graph.add_message(role="assistant", content="Hi there!")
    return graph


@pytest.fixture
def semantic_index():
    """Create test semantic index."""
    from context_core.testing import MockEmbeddingModel
    return SemanticIndex(MockEmbeddingModel(dimension=384))


@pytest.fixture
def token_budget():
    """Create test token budget."""
    return TokenBudget(total_tokens=10000)


@pytest.fixture
def memory_store(tmp_path):
    """Create test file system store."""
    return FileSystemStore(tmp_path)
```

---

## Configuration Contract

All packages accept configuration via Pydantic models:

```python
# context_engine/config.py

class CoreConfig(BaseModel):
    """Configuration for context-core."""
    model: str = "gpt-4"
    max_tokens: int = 128000


class CompressionConfig(BaseModel):
    """Configuration for context-compression."""
    strategies: list[str] = ["externalize", "dedupe", "schema"]
    preserve_recent: int = 10
    enable_recovery: bool = True


class MemoryConfig(BaseModel):
    """Configuration for context-memory."""
    backend: str = "filesystem"  # "filesystem", "postgres", "redis"
    connection_string: Optional[str] = None
    tiered: bool = False


class ToolsConfig(BaseModel):
    """Configuration for context-tools."""
    cache_enabled: bool = True
    cache_max_tokens: int = 100000
    prefetch_enabled: bool = False


class ObserveConfig(BaseModel):
    """Configuration for context-observe."""
    tracing: bool = True
    metrics: bool = True
    service_name: str = "context-engine"
    otlp_endpoint: Optional[str] = None


class MultiAgentConfig(BaseModel):
    """Configuration for context-multiagent."""
    transport: str = "memory"  # "memory", "redis", "kafka"
    sync_interval_seconds: float = 1.0


class ContextConfig(BaseModel):
    """Complete configuration."""
    core: CoreConfig = CoreConfig()
    compression: CompressionConfig = CompressionConfig()
    memory: MemoryConfig = MemoryConfig()
    tools: ToolsConfig = ToolsConfig()
    observe: ObserveConfig = ObserveConfig()
    multiagent: MultiAgentConfig = MultiAgentConfig()
```

---

*This API contract document ensures all packages can be developed independently while maintaining compatibility. Update this document when changing any public interface.*
