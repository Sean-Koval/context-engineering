# Competitive Analysis: Memori vs ContextEngine

> **Document Status**: Draft - iterating on analysis  
> **Last Updated**: 2026-01-12  
> **Repository**: https://github.com/MemoriLabs/Memori

## Executive Summary

| Aspect | **Memori** | **ContextEngine** |
|--------|-----------|-------------------|
| **Core Philosophy** | SQL-native memory layer with background enrichment | Graph-based context with intelligent compression |
| **Architecture** | Dual-process (immediate write + async enrichment) | Single-process with tiered compression pipeline |
| **Data Model** | Relational tables + knowledge graph triples | NetworkX graph with typed nodes/edges |
| **Processing Model** | Background thread pool for "augmentation" | In-band compression when budget thresholds trigger |
| **Primary Problem** | Persistent memory for facts/relationships | Context degradation at 60-70% capacity ("pre-rot") |
| **Target Use Case** | Cross-session memory persistence | In-session context optimization |

---

## Overview

### What is Memori?

Memori is "the memory fabric for enterprise AI" - a Python-based memory management system for LLMs, AI agents, and multi-agent systems. Key characteristics:

- **11.6k GitHub stars** (as of 2026-01)
- **Apache 2.0 license**
- **Enterprise-focused**: SQL-native with support for Postgres, MySQL, SQLite, MongoDB, Oracle
- **LLM-agnostic**: Works with OpenAI, Anthropic, Gemini, Bedrock, xAI
- **Framework integrations**: LangChain, Agno

### Core Insight Difference

**Memori's thesis**: 
> "Store conversations immediately, enrich them asynchronously in background threads, recall via semantic search"

**ContextEngine's thesis**:
> "Context should be a graph, not a string. Compress intelligently (compaction before summarization) to prevent pre-rot quality degradation"

---

## Architecture Comparison

### Memori's Dual-Process Design

```
User Interaction → Immediate Write to DB → Background Augmentation Thread
                        ↓                           ↓
                  Raw Conversations          Enriched Memories
                  (fast, zero latency)       (facts, relationships, entities)
```

**Key Components:**
- `Writer`: Immediate persistence with retry logic and transaction management
- `Collector`: Fire-and-forget payload submission with error recovery
- `AugmentationRuntime`: Daemon thread running async event loop
- `Manager`: Orchestrates augmentation with semaphore-limited concurrency (50 workers default)
- `Recall`: Embedding-based semantic search over stored facts

**Threading Model:**
```python
# Memori creates a dedicated daemon thread for async work
class AugmentationRuntime:
    - Daemon thread named 'memori-augmentation'
    - asyncio event loop running indefinitely
    - Semaphore limiting concurrent operations (default: 50)
    - Thread-safe initialization via locking
```

### ContextEngine's Pipeline Design

```
LLM Context → ContextGraph → Budget Monitor → Compression Pipeline
                  ↓                                    ↓
           Typed nodes/edges              Lossless → Compaction → Summarization
           Entity tracking                 (with RecoveryManifest)
           Semantic indexing
```

**Key Components:**
- `ContextGraph`: NetworkX-based graph with typed nodes/edges
- `TokenBudget`: Pre-rot detection with warning/trigger thresholds
- `CompressionPipeline`: Multi-strategy orchestration with tier ordering
- `RecoveryManifest`: Tracks operations for potential reconstruction
- `SemanticIndex`: Vector-based similarity search

---

## Feature Matrix

### Memory & Storage

| Feature | Memori | ContextEngine |
|---------|--------|---------------|
| **Memory Hierarchy** | Entity → Process → Session (3 levels) | Single graph with compression levels (4: FULL → EVICTED) |
| **Primary Storage** | SQL databases (Postgres, MySQL, SQLite, etc.) | In-memory graph + pluggable backends (planned) |
| **Knowledge Representation** | Semantic triples for relationships | Graph edges with typed relationships |
| **Schema Management** | Auto-migrations, 3NF design | Pydantic models, serialization to dict/JSON |
| **Vector Storage** | In-memory semantic search | Pluggable vector stores (memory, Qdrant) |

### Processing & Performance

| Feature | Memori | ContextEngine |
|---------|--------|---------------|
| **Background Processing** | Daemon thread with async event loop | None - compression is on-demand |
| **Concurrency Control** | Semaphore-limited (50 workers) | Single-threaded pipeline execution |
| **Write Strategy** | Batch writer (100 items, 0.1s timeout) | Direct graph mutations |
| **Latency Impact** | Zero - enrichment is async | Blocking during compression |

### Context Management

| Feature | Memori | ContextEngine |
|---------|--------|---------------|
| **Context Health Monitoring** | None | Pre-rot detection (warning: 0.5, trigger: 0.65) |
| **Compression** | None (append-only) | 3-tier pipeline with 8+ strategies |
| **Recovery** | N/A | RecoveryManifest for reconstruction |
| **Preservation Rules** | None | Recent nodes, pinned, high-importance, custom predicates |

### Entity & Fact Extraction

| Feature | Memori | ContextEngine |
|---------|--------|---------------|
| **Entity Extraction** | LLM-based "Advanced Augmentation" | spaCy NER + custom patterns |
| **Fact Extraction** | Extracts attributes, events, facts, preferences, relationships | Entity tracking only (no fact extraction) |
| **Relationship Discovery** | Automatic via augmentation | Manual edge creation |

### Observability

| Feature | Memori | ContextEngine |
|---------|--------|---------------|
| **Tracing** | Basic logging | OpenTelemetry native |
| **Metrics** | None | Prometheus/OTEL metrics |
| **Events** | None | Structured logging via structlog |

---

## What Memori Does That We Don't

### 1. Background Enrichment Thread

Their `AugmentationRuntime` processes data asynchronously with zero latency impact:

```python
# Their pattern (conceptual)
class AugmentationRuntime:
    def __init__(self, max_workers: int = 50):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore
        self._started: bool = False
        self._lock = threading.Lock()
    
    def ensure_started(self):
        with self._lock:
            if not self._started:
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name='memori-augmentation',
                    daemon=True
                )
                self._thread.start()
```

**Implication for ContextEngine**: We could add an async mode for compression that doesn't block the main flow.

### 2. SQL-Native Storage with Auto-Migrations

They support enterprise databases out of the box with automatic schema management. Our `context-memory` package spec describes this but isn't implemented.

**Implication**: Prioritize SQL backends in context-memory implementation.

### 3. Hierarchical Attribution (Entity → Process → Session)

```
Entity (user, organization)
    └── Process (agent, program)
            └── Session (current interaction)
```

This enables queries like "What does this user prefer across all their agent interactions?"

**Implication**: Enhance `context-multiagent` with attribution hierarchy.

### 4. Structured Fact Extraction

Their "Advanced Augmentation" extracts:
- **Attributes**: User characteristics
- **Events**: Things that happened  
- **Facts**: "User's favorite color is blue"
- **Preferences**: User likes/dislikes
- **Relationships**: "User knows person X"
- **Rules**: Business logic learned from interactions
- **Skills**: User capabilities

**Implication**: Extend `EntityTracker` to extract relationships and facts, not just entities.

### 5. Automatic Conversation Capture

They intercept LLM calls and store conversations automatically. We require explicit `add_message()` calls.

**Implication**: Consider middleware/callback integration for automatic capture.

### 6. Write Batching

Their `_db_writer` coalesces writes:
- Batch size: 100
- Timeout: 0.1s
- Reduces I/O for high-frequency operations

**Implication**: Add write coalescing to memory backends.

---

## What We Do That Memori Doesn't

### 1. Pre-Rot Detection

Our `TokenBudget` with warning/trigger thresholds addresses quality degradation **before** context limits hit:

```python
class TokenBudget:
    warning_threshold: float = 0.5   # Start monitoring
    trigger_threshold: float = 0.65  # Initiate compression
```

Memori has no concept of context window health or proactive management.

**This is a key differentiator** - we prevent degradation rather than just store more data.

### 2. Reversible Compression with Recovery

Our 3-tier compression pipeline preserves information:

```
LOSSLESS (zero loss)
    ├── ExternalizePayloads
    ├── DeduplicateSemantically  
    └── CollapseToolChains

COMPACTION (reversible)
    ├── SchemaCompression
    ├── EntityCentricCompression
    └── TaskRelevanceCompression

SUMMARIZATION (last resort)
    ├── HierarchicalSummarization
    ├── TaskAwareSummarization
    └── IncrementalSummarization
```

`RecoveryManifest` tracks all operations for potential reconstruction.

Memori's model is append-only - they enrich but never compress.

### 3. Graph-Based Context Representation

Our `ContextGraph` enables relationship queries:

```python
# Get related context
neighbors = graph.get_neighbors(node_id, edge_types=[EdgeType.CAUSAL])
subgraph = graph.get_subgraph(root_ids, max_depth=2)

# Query by entity
related = graph.get_by_entity("user-123")
```

Memori's "knowledge graph" is just storage triples, not queryable graph structures.

### 4. Preservation Rules

Fine-grained control over what gets compressed:

```python
PreservationRule(
    name="recent_messages",
    max_age_seconds=300,  # Last 5 minutes
)
PreservationRule(
    name="pinned_content",
    pinned=True,
)
PreservationRule(
    name="high_importance", 
    min_importance=0.9,
)
```

Memori has no selective compression - everything is stored equally.

### 5. Token-Aware Budget Management

Section-based allocation with overflow handling:

```python
budget = TokenBudget(
    total_tokens=200_000,
    sections={
        "system": 0.1,
        "history": 0.6,
        "tools": 0.3
    }
)
```

Memori ignores token limits entirely.

### 6. OpenTelemetry Native Observability

Full tracing, metrics, and structured logging:
- Span creation for all context operations
- Prometheus/OTEL metrics export
- Compression events, pre-rot triggers

Memori has basic Python logging only.

### 7. Tool Call Optimization (Planned)

Our `context-tools` spec includes:
- `ToolCallCache`: Semantic caching for tool results
- `ToolUsagePatterns`: Learn from usage history
- `ToolResultCompressor`: Specialized result processing
- `ToolPrefetcher`: Predictive execution

Memori doesn't optimize tool interactions.

---

## Design Philosophy Contrast

### Memori's Approach

> "Store everything immediately, enrich asynchronously, recall semantically"

**Strengths:**
- Zero latency impact on user interactions
- Enterprise database compatibility
- LLM-powered fact extraction is powerful
- Good for cross-session memory

**Weaknesses:**
- No context window management
- No compression = unbounded growth
- No quality degradation prevention
- Recall depends on embedding quality

### ContextEngine's Approach

> "Context is a graph. Compress intelligently before summarizing. Prevent pre-rot."

**Strengths:**
- Proactive quality management
- Information preservation through reversible compression
- Rich graph operations for relationship queries
- Token-aware with budget allocation

**Weaknesses:**
- Currently synchronous/blocking
- No automatic conversation capture
- Memory backends not yet implemented
- No cross-session persistence (yet)

---

## Opportunities for Adoption

### High Value - Should Implement

#### 1. Background Processing Runtime

Adapt their `AugmentationRuntime` pattern for async compression:

```python
class CompressionRuntime:
    """Background thread for async compression operations."""
    
    def __init__(self, max_workers: int = 10):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore
        
    def schedule_compression(self, graph: ContextGraph, target: int) -> Future:
        """Schedule compression without blocking main flow."""
        return asyncio.run_coroutine_threadsafe(
            self._compress_async(graph, target),
            self._loop
        )
```

**Benefit**: Zero-latency compression when budget crosses warning threshold.

#### 2. Attribution Hierarchy for Multi-Agent

Adopt their Entity → Process → Session model:

```python
class Attribution(BaseModel):
    entity_id: str      # User or organization
    process_id: str     # Agent or workflow  
    session_id: str     # Current interaction
    
class ContextNode:
    attribution: Attribution  # Add to existing model
```

**Benefit**: Better context sharing and ownership in `context-multiagent`.

#### 3. Write Batching Layer

Add coalescing for memory backends:

```python
class BatchWriter:
    def __init__(self, batch_size: int = 100, timeout: float = 0.1):
        self._queue: asyncio.Queue
        self._batch_size = batch_size
        self._timeout = timeout
        
    async def enqueue(self, operation: WriteOp) -> None:
        await self._queue.put(operation)
        
    async def _flush_loop(self) -> None:
        # Batch and flush periodically
```

**Benefit**: Reduced I/O for high-frequency tool calls.

### Medium Value - Consider Implementing

#### 4. Structured Fact Extraction

Extend `EntityTracker` to extract relationships and facts:

```python
class FactExtractor:
    def extract(self, text: str) -> list[Fact]:
        """Extract structured facts from text."""
        # Could use LLM or rule-based extraction
        
class Fact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float
    source_node_id: UUID
```

**Benefit**: Richer entity-centric compression, better recall.

#### 5. Automatic Conversation Capture

Middleware for popular frameworks:

```python
class ContextEngineCallback(LangChainCallback):
    def on_llm_start(self, prompts, **kwargs):
        for prompt in prompts:
            self.graph.add_message("user", prompt)
            
    def on_llm_end(self, response, **kwargs):
        self.graph.add_message("assistant", response.text)
```

**Benefit**: Easier integration, less manual instrumentation.

### Lower Priority - Nice to Have

#### 6. SQL-Native Storage

Already in `context-memory` spec, but could prioritize:
- PostgreSQL with pgvector
- SQLite for local development
- Auto-migrations

#### 7. Knowledge Graph Triples

Store semantic triples alongside graph edges:

```python
class Triple(BaseModel):
    subject: str
    predicate: str  
    object: str
    source_node_ids: list[UUID]
```

---

## Strategic Conclusions

### Complementary, Not Competing

The projects solve **different problems**:

| Problem Space | Memori | ContextEngine |
|--------------|--------|---------------|
| Cross-session memory | ✅ Primary focus | ❌ Not addressed |
| In-session optimization | ❌ Not addressed | ✅ Primary focus |
| Pre-rot prevention | ❌ | ✅ |
| Fact extraction | ✅ | ❌ |
| Compression | ❌ | ✅ |
| Enterprise SQL | ✅ | 🔄 Planned |

### Potential Integration

ContextEngine + Memori could work together:

```
Session Start
    ↓
Memori recalls relevant facts → Seed ContextGraph
    ↓
ContextEngine manages in-session context
    ↓
Session End
    ↓
ContextEngine exports summaries → Memori stores for future recall
```

### Key Takeaways

1. **Our pre-rot detection is unique** - Memori should steal this from us
2. **Their background processing is elegant** - We should adopt this pattern
3. **Our compression is more sophisticated** - They have no equivalent
4. **Their enterprise focus is mature** - We need SQL backends sooner
5. **Graph vs. triples**: Our approach is more powerful for relationship traversal
6. **Attribution hierarchy**: Good pattern for multi-agent work

### Recommended Actions

1. **Immediate**: Add async compression option with background thread
2. **Phase 3**: Prioritize PostgreSQL backend in context-memory
3. **Phase 4**: Add attribution hierarchy to context-multiagent
4. **Research**: Evaluate LLM-based fact extraction vs. NER-only approach

---

## Appendix: Code References

### Memori Repository Structure

```
memori/
├── memory/
│   ├── _collector.py      # Fire-and-forget payload submission
│   ├── _manager.py        # Memory operation orchestration
│   ├── _struct.py         # Data structures
│   ├── _writer.py         # Database persistence
│   ├── recall.py          # Semantic retrieval
│   └── augmentation/
│       ├── _runtime.py    # Background thread management
│       ├── _manager.py    # Augmentation orchestration
│       ├── _db_writer.py  # Batch database writes
│       └── augmentations/ # Enrichment implementations
├── storage/
│   ├── _base.py           # Storage abstractions
│   ├── _builder.py        # Builder pattern for config
│   ├── adapters/          # ORM adapters
│   └── drivers/           # Database drivers
└── llm/                   # LLM provider integrations
```

### ContextEngine Relevant Files

- `packages/context-core/src/context_core/graph/context_graph.py` - Graph implementation
- `packages/context-core/src/context_core/budget/token_budget.py` - Pre-rot detection
- `packages/context-compression/src/context_compression/pipeline.py` - Compression orchestration
- `packages/context-compression/src/context_compression/recovery/` - Recovery manifest

---

*Document maintained by the ContextEngine team. Contributions welcome.*
