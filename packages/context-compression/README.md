# context-compression

> Intelligent compression pipeline with 9 tiered strategies for LLM context management

## Installation

```bash
uv pip install -e packages/context-core -e packages/context-compression
```

## Overview

The compression pipeline applies strategies in order of information preservation:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPRESSION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TIER 1: LOSSLESS (100% recoverable)                            │
│  ├── ExternalizePayloads      → Move large outputs externally   │
│  ├── DeduplicateSemantically  → Remove near-duplicates          │
│  └── CollapseToolChains       → Merge sequential tool calls     │
│                                                                  │
│  TIER 2: COMPACTION (structure preserved)                       │
│  ├── SchemaCompression        → Extract repeated JSON schemas   │
│  ├── EntityCentricCompression → Keep entity-relevant content    │
│  └── TaskRelevanceCompression → Filter by task relevance        │
│                                                                  │
│  TIER 3: SUMMARIZATION (last resort)                            │
│  ├── HierarchicalSummarization → Bottom-up multi-level          │
│  ├── TaskAwareSummarization    → Task-focused summaries         │
│  └── IncrementalSummarization  → Streaming updates              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
from context_core import ContextGraph
from context_compression import CompressionPipeline, CompressionTier

# Create a graph with content
graph = ContextGraph(session_id="demo")
# ... add nodes ...

# Create pipeline with default strategies
pipeline = CompressionPipeline()

# Preview compression (dry-run)
plan = pipeline.plan(graph)
print(f"Would save {plan.estimated_savings} tokens")

# Apply compression up to COMPACTION tier
results = pipeline.compress(
    graph,
    max_tier=CompressionTier.COMPACTION
)

for result in results:
    print(f"{result.strategy_name}: saved {result.tokens_saved} tokens")

# Compress to specific budget
results = pipeline.compress_to_budget(
    graph,
    target_tokens=50_000
)
```

## Strategies

### Tier 1: Lossless Strategies

#### ExternalizePayloads

Moves large outputs (tool results, artifacts) to external storage, keeping only a reference.

```python
from context_compression.strategies import ExternalizePayloads
from context_compression.strategies.lossless import (
    InMemoryExternalStorage,
    FileSystemExternalStorage,
)

# With in-memory storage (development)
strategy = ExternalizePayloads(
    storage=InMemoryExternalStorage(),
    min_size_bytes=1000,  # Only externalize content > 1KB
)

# With filesystem storage (production)
strategy = ExternalizePayloads(
    storage=FileSystemExternalStorage(base_path="/tmp/context-storage"),
    min_size_bytes=1000,
)

# Apply to graph
result = strategy.compress(graph, manifest)
```

**Compression**: 2-5x for tool-heavy contexts

#### DeduplicateSemantically

Removes semantically similar content, keeping only the canonical version.

```python
from context_compression.strategies import DeduplicateSemantically
from context_core import SemanticIndex

strategy = DeduplicateSemantically(
    semantic_index=SemanticIndex(),
    similarity_threshold=0.92,  # 92% similar = duplicate
)

result = strategy.compress(graph, manifest)
```

**Compression**: 1.5-2x for repetitive contexts

#### CollapseToolChains

Merges sequential related tool calls into a single summary node.

```python
from context_compression.strategies import CollapseToolChains

strategy = CollapseToolChains(
    min_chain_length=3,    # Minimum calls to collapse
    max_time_gap=60.0,     # Max seconds between calls
)

result = strategy.compress(graph, manifest)
```

**Compression**: 2-3x for tool-heavy workflows

### Tier 2: Compaction Strategies

#### SchemaCompression

Extracts common JSON schemas from arrays of objects.

```python
from context_compression.strategies import SchemaCompression

strategy = SchemaCompression(
    min_occurrences=2,     # Minimum times schema appears
    min_array_length=3,    # Minimum array size to compress
)

# Before: {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
# After:  {"users": {"$schema_ref": "abc", "keys": ["id", "name"], "values": [[1, "Alice"], [2, "Bob"]]}}

result = strategy.compress(graph, manifest)
```

**Compression**: 2-4x for structured JSON outputs

#### EntityCentricCompression

Keeps only sentences mentioning important entities.

```python
from context_compression.strategies import EntityCentricCompression
from context_core import EntityTracker

tracker = EntityTracker()
# ... extract entities from context ...

strategy = EntityCentricCompression(
    entity_tracker=tracker,
    min_importance=0.3,  # Only keep entities with importance > 0.3
)

result = strategy.compress(graph, manifest)
```

**Compression**: 2-3x for verbose content

#### TaskRelevanceCompression

Filters content based on relevance to the current task.

```python
from context_compression.strategies import TaskRelevanceCompression

strategy = TaskRelevanceCompression(
    relevance_threshold=0.3,
    task_context_messages=5,  # Last N messages define task
)

result = strategy.compress(graph, manifest)
```

**Compression**: 2-4x depending on task focus

### Tier 3: Summarization Strategies

All summarization strategies require an `LLMSummarizer`:

```python
from context_compression.strategies.summarization import (
    LLMSummarizer,
    MockLLMSummarizer,  # For testing
)

# For testing (deterministic, no API calls)
summarizer = MockLLMSummarizer(compression_ratio=0.2)

# For production, implement LLMSummarizer protocol
class OpenAISummarizer:
    def summarize(self, texts: list[str], max_tokens: int, ...) -> str:
        # Call OpenAI API
        ...
```

#### HierarchicalSummarization

Bottom-up multi-level summarization.

```python
from context_compression.strategies import HierarchicalSummarization

strategy = HierarchicalSummarization(
    summarizer=summarizer,
    chunk_size=10,              # Messages per chunk
    chunk_token_threshold=2000, # Token trigger
    preserve_recent_chunks=1,   # Keep recent unsummarized
)

result = strategy.compress(graph, manifest)
```

**Compression**: 5-10x

#### TaskAwareSummarization

Task-focused summaries that preserve relevant context.

```python
from context_compression.strategies import TaskAwareSummarization

strategy = TaskAwareSummarization(
    summarizer=summarizer,
    task_context_messages=5,
    relevance_threshold=0.3,
)

result = strategy.compress(graph, manifest)
```

**Compression**: 5-10x with better task preservation

#### IncrementalSummarization

Streaming summarization with a running summary.

```python
from context_compression.strategies import IncrementalSummarization

strategy = IncrementalSummarization(
    summarizer=summarizer,
    update_interval=5,       # Messages before update
    max_summary_tokens=500,  # Summary size cap
)

result = strategy.compress(graph, manifest)
```

**Compression**: 5-10x with lower latency

## Recovery Manifest

All operations are logged for potential recovery:

```python
from context_compression import RecoveryManifest

manifest = RecoveryManifest()

# After compression
results = pipeline.compress(graph, manifest=manifest)

# Check what operations occurred
for op in manifest.operations:
    print(f"{op.op_type}: node {op.node_id}")

# Check if a node can be recovered
can_recover = manifest.can_recover_node(node_id)

# Get recovery instructions
instructions = manifest.get_recovery_instructions(node_id)

# Statistics
stats = manifest.stats()
print(f"Total operations: {stats.total_operations}")
print(f"Recoverable: {stats.recoverable_operations}")
```

### Operation Types

| Operation | Recoverable | Description |
|-----------|-------------|-------------|
| `ExternalizeOperation` | ✅ Yes | Content stored externally |
| `DeduplicateOperation` | ✅ Yes | Original content preserved |
| `CollapseOperation` | ✅ Yes | Original sequence saved |
| `CompactOperation` | ⚠️ Partial | Structure known, some data lost |
| `SummarizeOperation` | ❌ No | Original content lost |
| `EvictOperation` | ⚠️ Depends | Recoverable if externalized first |

## Custom Strategies

Create custom strategies by implementing the protocol:

```python
from context_compression import CompressionStrategy, CompressionTier, CompressionResult
from context_compression.strategies import BaseCompressionStrategy

class MyCustomStrategy(BaseCompressionStrategy):
    """Custom compression strategy."""

    @property
    def _name(self) -> str:
        return "my_custom_strategy"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def _priority(self) -> int:
        return 25  # Run after schema compression (20)

    def _can_apply_impl(self, graph) -> bool:
        # Check if strategy is applicable
        return len(graph) > 0

    def _estimate_savings_impl(self, graph, target_node_ids) -> int:
        # Estimate tokens that would be saved
        return 1000

    def _compress_impl(self, graph, manifest, target_node_ids, target_tokens):
        # Implement compression logic
        # Log operations to manifest
        # Return CompressionResult
        ...

# Register with pipeline
pipeline = CompressionPipeline()
pipeline.register_strategy(MyCustomStrategy())
```

## Preservation Rules

Protect important nodes from compression:

```python
from context_compression import CompressionPipeline, PreservationRule

pipeline = CompressionPipeline(
    preservation_rules=[
        PreservationRule(
            name="protect_recent",
            condition=lambda n: n.metadata.sequence > graph.max_sequence - 5,
            reason="Recent messages",
        ),
        PreservationRule(
            name="protect_pinned",
            condition=lambda n: n.metadata.pinned,
            reason="User-pinned content",
        ),
        PreservationRule(
            name="protect_high_importance",
            condition=lambda n: n.compute_importance() > 0.9,
            reason="High importance",
        ),
    ]
)
```

## Testing

```bash
# Run all compression tests
uv run pytest packages/context-compression/tests/ -v

# Run specific strategy tests
uv run pytest packages/context-compression/tests/strategies/test_lossless.py
uv run pytest packages/context-compression/tests/strategies/test_compaction.py
uv run pytest packages/context-compression/tests/strategies/test_summarization.py

# With coverage
uv run pytest packages/context-compression/tests/ --cov=context_compression
```

## Performance

| Strategy | Compression | Latency | Recoverability |
|----------|-------------|---------|----------------|
| ExternalizePayloads | 2-5x | <10ms | 100% |
| DeduplicateSemantically | 1.5-2x | ~50ms | 100% |
| CollapseToolChains | 2-3x | <10ms | 100% |
| SchemaCompression | 2-4x | <20ms | 80-95% |
| EntityCentricCompression | 2-3x | ~30ms | 80-95% |
| TaskRelevanceCompression | 2-4x | ~30ms | 80-95% |
| HierarchicalSummarization | 5-10x | ~500ms* | 0% |
| TaskAwareSummarization | 5-10x | ~500ms* | 0% |
| IncrementalSummarization | 5-10x | ~200ms* | 0% |

*Latency depends on LLM backend
