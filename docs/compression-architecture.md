# Context Compression Pipeline - Architecture Analysis

> Prepared by: Claude Agent
> Date: 2026-01-10
> Purpose: Evaluate components and establish implementation approach

---

## 1. Integration Points with context-core

### 1.1 Types Already Available

| Type | Location | Usage in Compression |
|------|----------|---------------------|
| `CompressionLevel` | `graph/types.py` | FULL→COMPACTED→SUMMARIZED→EVICTED |
| `NodeType` | `graph/types.py` | Filter strategies by node type |
| `EdgeType` | `graph/types.py` | SUMMARIZES edge for linking |
| `ContextNode` | `graph/nodes.py` | Primary compression target |
| `Content` | `graph/nodes.py` | Has `external_ref`, `original_tokens`, etc. |
| `NodeMetadata` | `graph/nodes.py` | Has `pinned`, `preserve_until`, `min_compression_level` |

### 1.2 Key Context-Core Interfaces

```python
# ContextGraph methods we'll use
graph.get_recent(n)         # For preserve_recent_n
graph.query_nodes(...)      # For finding compression candidates
graph.get_neighbors(id)     # For understanding relationships
graph.get_subgraph(ids)     # For extracting portions to summarize
graph.update_node(node)     # For updating compression_level
graph.remove_node(id)       # For eviction

# ContextNode properties we'll use
node.compute_importance()   # For prioritization
node.compression_level      # For tracking state
node.token_count           # For budget calculations
node.metadata.pinned       # For preservation
node.metadata.preserve_until  # For temporal preservation

# TokenBudget integration
budget.is_pre_rot_trigger   # When to start compression
budget.get_compression_priority()  # Which sections first
budget.deallocate(section, tokens)  # After compression
```

### 1.3 EntityTracker Integration

For `EntityCentricCompression`:
```python
tracker.get_entities_by_node(node_id)  # Find entities in node
tracker.get_top_entities(n)            # Most important entities
entity.importance                       # For preservation decisions
```

### 1.4 SemanticIndex Integration (Optional Dependency)

For `DeduplicateSemantically`:
```python
index.find_similar(text, threshold)  # Find semantic duplicates
index.add_embedding(node_id, text)   # Index for deduplication
```

**Note**: SemanticIndex is being implemented by another agent. DeduplicateSemantically should be designed to work optionally or gracefully degrade.

---

## 2. Component Dependency Map

```
                    ┌─────────────────────────────────────────┐
                    │           context-compression           │
                    └─────────────────────────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
    ┌───────────────┐         ┌──────────────────┐         ┌──────────────────┐
    │ CompressionPipeline │◄──│  CompressionStrategy  │     │ RecoveryManifest │
    └───────────────┘         └──────────────────┘         └──────────────────┘
            │                           │                           │
            │                  ┌────────┼────────┐                  │
            │                  │        │        │                  │
            ▼                  ▼        ▼        ▼                  ▼
    ┌───────────────┐    ┌─────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐
    │PreservationRule│    │Lossless│ │Compaction│ │Summarize│ │OperationType│
    └───────────────┘    └─────────┘ └────────┘ └──────────┘ └────────────┘
                              │          │           │
                              ▼          ▼           ▼
                    ┌──────────────────────────────────────────┐
                    │              context-core                │
                    │  (ContextGraph, ContextNode, TokenBudget)│
                    └──────────────────────────────────────────┘
```

---

## 3. Strategy Dependencies

| Strategy | Tier | Dependencies | Can Build Now? |
|----------|------|--------------|----------------|
| **ExternalizePayloads** | LOSSLESS | ExternalStorage protocol | ✅ Yes |
| **DeduplicateSemantically** | LOSSLESS | SemanticIndex | ⚠️ Optional |
| **CollapseToolChains** | LOSSLESS | ContextGraph (edges) | ✅ Yes |
| **SchemaCompression** | COMPACTION | None | ✅ Yes |
| **EntityCentricCompression** | COMPACTION | EntityTracker | ✅ Yes |
| **TaskRelevanceCompression** | COMPACTION | None | ✅ Yes |
| **HierarchicalSummarization** | SUMMARIZATION | LLMSummarizer protocol | ⚠️ Protocol |
| **TaskAwareSummarization** | SUMMARIZATION | LLMSummarizer protocol | ⚠️ Protocol |
| **IncrementalSummarization** | SUMMARIZATION | LLMSummarizer protocol | ⚠️ Protocol |

---

## 4. Recommended Implementation Order

### Phase 2A: Foundation (Can start immediately)

1. **Core Types & Models** (CP-001 to CP-003)
   - `CompressionTier` enum
   - `CompressionResult` model
   - `PreservationRule` model
   - `CompressionPlan` model

2. **Strategy Protocol** (CP-004)
   - `CompressionStrategy` protocol definition
   - Abstract base with common functionality

3. **CompressionPipeline** (CP-005 to CP-007)
   - Strategy registration
   - Ordering by tier and priority
   - Preservation logic
   - Core compress() method
   - Dry-run preview capability

### Phase 2B: Lossless Strategies

4. **ExternalStorage Protocol** (LS-001, LS-002)
   - Protocol definition
   - InMemoryExternalStorage reference implementation

5. **ExternalizePayloads** (LS-003)
   - Identifies large payloads
   - Stores externally, keeps preview
   - Fully reversible

6. **CollapseToolChains** (LS-005)
   - Identifies sequential tool calls
   - Collapses to summary with before/after

7. **DeduplicateSemantically** (LS-004)
   - Optional SemanticIndex dependency
   - Graceful degradation if unavailable
   - Keeps one, removes duplicates

### Phase 2C: Compaction Strategies

8. **SchemaCompression** (CS-001)
   - Extract JSON schemas from tool results
   - Replace data with schema + sample

9. **EntityCentricCompression** (CS-002)
   - Use EntityTracker to identify key entities
   - Preserve entity context, compress rest

10. **TaskRelevanceCompression** (CS-003)
    - Score relevance to current task
    - Remove off-task content

### Phase 2D: Recovery & Summarization

11. **Operation Types** (SS-004)
    - All operation dataclasses
    - is_recoverable logic

12. **RecoveryManifest** (SS-005)
    - Operation logging
    - Recovery capability checking
    - Statistics

13. **LLMSummarizer Protocol** (SS-001)
    - Abstract protocol
    - MockSummarizer for testing

14. **Summarization Strategies** (SS-002, SS-003)
    - HierarchicalSummarization
    - TaskAwareSummarization

### Phase 2E: Integration

15. **Public API** (INT-001)
    - Clean exports in `__init__.py`
    - High-level convenience functions

16. **Integration Tests** (INT-002)
    - Full pipeline tests
    - Cross-package integration

---

## 5. Key Architecture Decisions

### 5.1 Strategy Interface: Protocol vs ABC

**Decision**: Use `Protocol` with optional ABC for shared functionality

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class CompressionStrategy(Protocol):
    """Protocol for compression strategies."""

    @property
    def name(self) -> str: ...

    @property
    def tier(self) -> CompressionTier: ...

    def can_compress(self, node: ContextNode) -> bool: ...

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        preserved: set[UUID] | None = None,
    ) -> CompressionResult: ...
```

**Rationale**: Protocol allows flexibility; implementations don't need to inherit. ABC can provide shared helpers.

### 5.2 Optional Dependencies

**Decision**: Use optional imports with graceful degradation

```python
try:
    from context_core.semantic import SemanticIndex
    HAS_SEMANTIC_INDEX = True
except ImportError:
    HAS_SEMANTIC_INDEX = False

class DeduplicateSemantically:
    def __init__(self, semantic_index: SemanticIndex | None = None):
        if semantic_index is None and not HAS_SEMANTIC_INDEX:
            raise RuntimeError("SemanticIndex required for semantic deduplication")
        self._index = semantic_index
```

### 5.3 RecoveryManifest Persistence

**Decision**: Serialize with graph, allow separate storage

```python
class RecoveryManifest:
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryManifest: ...

    # Optional: save to external storage
    async def persist(self, storage: ExternalStorage) -> str: ...
```

### 5.4 Pipeline Execution Model

**Decision**: Iterative compression with early stopping

```python
def compress(
    self,
    graph: ContextGraph,
    target_tokens: int | None = None,
    max_iterations: int = 10,
) -> list[CompressionResult]:
    """
    Compress until target_tokens reached or no more progress.

    - Strategies executed in tier order (LOSSLESS → COMPACTION → SUMMARIZATION)
    - Within tier, ordered by priority
    - Stops when target reached or no progress in iteration
    """
```

### 5.5 Preservation Rules

**Decision**: Multiple rule types, evaluated in order

```python
class PreservationRule(BaseModel):
    name: str
    rule_type: Literal["pinned", "recent", "type", "entity", "importance", "custom"]
    # Type-specific config
    node_types: list[NodeType] | None = None
    recent_n: int | None = None
    min_importance: float | None = None
    entity_ids: set[UUID] | None = None
    custom_filter: Callable[[ContextNode], bool] | None = None
```

---

## 6. Package Structure

```
packages/context-compression/
├── pyproject.toml
├── src/
│   └── context_compression/
│       ├── __init__.py          # Public API exports
│       ├── types.py             # CompressionTier, CompressionResult, etc.
│       ├── pipeline.py          # CompressionPipeline
│       ├── preservation.py      # PreservationRule, PreservationEvaluator
│       ├── recovery.py          # RecoveryManifest, operations
│       └── strategies/
│           ├── __init__.py
│           ├── base.py          # Strategy protocol/ABC
│           ├── lossless/
│           │   ├── __init__.py
│           │   ├── externalize.py
│           │   ├── deduplicate.py
│           │   └── collapse.py
│           ├── compaction/
│           │   ├── __init__.py
│           │   ├── schema.py
│           │   ├── entity_centric.py
│           │   └── task_relevance.py
│           └── summarization/
│               ├── __init__.py
│               ├── protocols.py  # LLMSummarizer
│               ├── hierarchical.py
│               └── task_aware.py
└── tests/
    ├── __init__.py
    ├── test_types.py
    ├── test_pipeline.py
    ├── test_preservation.py
    ├── test_recovery.py
    └── strategies/
        ├── test_lossless.py
        ├── test_compaction.py
        └── test_summarization.py
```

---

## 7. Implementation Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| SemanticIndex not ready | DeduplicateSemantically blocked | Make it optional, provide stub |
| LLM summarization expensive | Slow compression | Provide mock, allow async |
| Token counting inconsistent | Budget miscalculations | Use context-core Tokenizer |
| Graph mutations during compression | Race conditions | Copy-on-write or lock |
| Large graphs | Memory pressure | Streaming/batched processing |

---

## 8. Success Criteria

- [ ] All types pass mypy strict
- [ ] CompressionPipeline handles edge cases
- [ ] Each strategy has 90%+ test coverage
- [ ] Integration tests pass with context-core
- [ ] Compression ratios meet targets
- [ ] Performance benchmarks established

---

## 9. Next Steps

1. **Create package structure** - Set up `context-compression` package
2. **Implement types.py** - Core types and models
3. **Implement base.py** - Strategy protocol
4. **Implement pipeline.py** - Core pipeline logic
5. **Add first strategy** - ExternalizePayloads (simplest lossless)
6. **Iterate** - Add strategies one by one with tests

---

## 10. Detailed Interface Designs

### 10.1 CompressionStrategy Protocol

```python
from typing import Protocol, runtime_checkable
from uuid import UUID

@runtime_checkable
class CompressionStrategy(Protocol):
    """Protocol for all compression strategies."""

    @property
    def name(self) -> str:
        """Unique strategy identifier (e.g., 'externalize_payloads')."""
        ...

    @property
    def tier(self) -> CompressionTier:
        """Compression tier: LOSSLESS, COMPACTION, or SUMMARIZATION."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority within tier (lower = run first)."""
        ...

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None = None,
    ) -> int:
        """Estimate tokens that would be saved without modifying graph."""
        ...

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None = None,
        target_tokens: int | None = None,
    ) -> CompressionResult:
        """Execute compression, modifying graph in place."""
        ...

    def can_apply(self, graph: ContextGraph) -> bool:
        """Check if this strategy can be applied to the graph."""
        ...
```

### 10.2 Core Types

```python
class CompressionTier(str, Enum):
    """Tier of compression strategy."""
    LOSSLESS = "lossless"      # Fully reversible
    COMPACTION = "compaction"   # Mostly reversible
    SUMMARIZATION = "summarization"  # Irreversible

class CompressionResult(BaseModel):
    """Result of a compression operation."""
    success: bool
    strategy_name: str
    tier: CompressionTier
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    nodes_processed: int
    nodes_compressed: int
    nodes_removed: int
    nodes_created: int
    duration_ms: float
    is_recoverable: bool
    manifest_id: UUID | None = None

    @property
    def compression_ratio(self) -> float:
        if self.compressed_tokens == 0:
            return float('inf')
        return self.original_tokens / self.compressed_tokens

class PreservationRule(BaseModel):
    """Rule for preserving nodes from compression."""
    name: str
    description: str
    priority: int = 0
    node_types: list[str] | None = None
    min_importance: float | None = None
    max_age_seconds: int | None = None
    required_tags: set[str] | None = None
    entity_ids: list[str] | None = None
    pinned: bool = False
```

### 10.3 RecoveryManifest Operations

```python
class ExternalizeOperation(BaseModel):
    """Tracks externalized content for recovery."""
    op_type: Literal["externalize"] = "externalize"
    node_id: UUID
    external_uri: str
    original_tokens: int
    original_content_hash: str
    preview: str | None = None

class DeduplicateOperation(BaseModel):
    """Tracks deduplicated nodes."""
    op_type: Literal["deduplicate"] = "deduplicate"
    removed_node_ids: list[UUID]
    kept_node_id: UUID
    similarity_score: float

class CollapseOperation(BaseModel):
    """Tracks collapsed tool chains."""
    op_type: Literal["collapse"] = "collapse"
    original_node_ids: list[UUID]
    collapsed_node_id: UUID
    chain_description: str

class SummarizeOperation(BaseModel):
    """Tracks summarization (not fully recoverable)."""
    op_type: Literal["summarize"] = "summarize"
    original_node_ids: list[UUID]
    summary_node_id: UUID
    original_tokens: int
    summary_tokens: int
    method: str

# Union type for all operations
CompressionOperation = (
    ExternalizeOperation
    | DeduplicateOperation
    | CollapseOperation
    | SummarizeOperation
)
```

---

## 11. Testing Strategy

### Unit Tests Required

| Component | Test Focus | Coverage Target |
|-----------|------------|-----------------|
| `types.py` | Model validation, computed properties | 100% |
| `pipeline.py` | Strategy ordering, preservation, execution | 95% |
| `preservation.py` | Rule matching, edge cases | 95% |
| `recovery.py` | Operation logging, recovery checks | 95% |
| Each strategy | Core functionality, edge cases | 90% |

### Integration Tests

1. **Pipeline + ContextGraph**: Full compression cycle
2. **Pipeline + TokenBudget**: Budget-aware compression
3. **Strategies + RecoveryManifest**: Recovery verification
4. **Cross-strategy**: Multiple strategies in sequence

### Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| 1,000 nodes compression | < 500ms |
| 10,000 nodes compression | < 5s |
| Recovery manifest lookup | < 10ms |
| Strategy estimation | < 100ms |

---

*This document provides the foundation for implementing a well-designed compression pipeline that integrates cleanly with context-core.*
