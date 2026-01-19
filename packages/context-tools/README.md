# context-tools

> Tool call caching, usage pattern detection, and predictive prefetching for ContextEngine

## Installation

```bash
uv pip install -e packages/context-tools
```

## Components

### ToolCallCache

Multi-strategy caching with exact, normalized, and semantic matching.

```python
from context_tools import ToolCallCache, ToolCallSignature

# Create cache with semantic matching
cache = ToolCallCache(
    max_entries=1000,
    max_tokens=100000,
    default_ttl_seconds=3600,
    semantic_threshold=0.85,
    enable_semantic=True,
)

# Create signature for a tool call
sig = ToolCallSignature(
    tool_name="read_file",
    arguments={"path": "/src/main.py"},
)

# Store result
await cache.put(sig, file_content, token_count=500)

# Retrieve (tries exact → normalized → semantic matching)
result = await cache.get(sig)
if result:
    print(f"Cache hit! Tokens saved: {result.result_tokens}")

# Check stats
print(f"Hit rate: {cache.stats.hit_rate:.1%}")
print(f"Semantic hits: {cache.stats.semantic_hits}")
```

### ToolUsagePatterns

Learn tool sequences and predict next likely calls.

```python
from context_tools import ToolUsagePatterns, ToolCallSignature

patterns = ToolUsagePatterns(
    min_pattern_frequency=3,
    max_history=1000,
)

# Record tool calls as they happen
patterns.record(ToolCallSignature(tool_name="glob", arguments={"pattern": "*.py"}))
patterns.record(ToolCallSignature(tool_name="read_file", arguments={"path": "main.py"}))
patterns.record(ToolCallSignature(tool_name="grep", arguments={"pattern": "def"}))

# Predict next tool
predictions = patterns.predict_next_tool("glob", top_k=3)
# [("read_file", 0.8), ("grep", 0.15), ("list_directory", 0.05)]

# Detect common sequences
sequences = patterns.detect_sequences(min_length=2)
# [ToolSequence(["glob", "read_file"], frequency=15, confidence=0.8)]

# Predict likely arguments
predicted_args = patterns.predict_arguments("read_file")
# {"path": "/src/main.py"}  (most common value)

# Detect antipatterns
antipatterns = patterns.detect_antipatterns()
# [Antipattern(type="repeated_calls", tool="read_file", count=5)]
```

### ToolResultCompressor

Intelligent compression of tool results with schema extraction.

```python
from context_tools import ToolResultCompressor, SchemaCache

# Create compressor with schema caching
schema_cache = SchemaCache(max_schemas=100)
compressor = ToolResultCompressor(
    list_truncate_threshold=20,
    list_keep_items=5,
    schema_cache=schema_cache,
    use_smart_truncation=True,
)

# Compress a tool result
result = compressor.compress("search_files", {
    "results": [
        {"path": "/src/a.py", "score": 0.95, "preview": "..."},
        {"path": "/src/b.py", "score": 0.90, "preview": "..."},
        # ... 100 more results
    ]
})

print(f"Compression ratio: {result.compression_ratio:.1f}x")
print(f"Strategies used: {result.metadata['strategies_used']}")
# Strategies: schema_extraction, list_truncation, low_value_filtering
```

### ListTruncator

Smart list truncation with multiple sampling strategies.

```python
from context_tools import ListTruncator, TruncationStrategy

truncator = ListTruncator(
    default_strategy=TruncationStrategy.HEAD_TAIL,
    include_statistics=True,
)

# Truncate a large list
data = list(range(1000))
result = truncator.truncate(data, keep=10)

print(f"Kept items: {result.items}")  # [0, 1, 2, 3, 4, 995, 996, 997, 998, 999]
print(f"Statistics: min={result.statistical_summary.min_value}, max={result.statistical_summary.max_value}")

# Available strategies
# - HEAD_TAIL: Keep first and last items
# - UNIFORM: Evenly spaced samples
# - RESERVOIR: Random sampling
# - DIVERSE: Maximize variety (for objects)
# - STRATIFIED: Preserve type distribution
```

### ToolPrefetcher

Predictive tool execution to reduce latency.

```python
from context_tools import ToolPrefetcher

# Create prefetcher with patterns and cache
prefetcher = ToolPrefetcher(
    patterns=patterns,
    cache=cache,
    tool_executor=execute_tool,  # Your async tool executor
    min_confidence=0.5,
    max_concurrent_prefetch=3,
)

# After a tool completes, trigger prefetching
await prefetcher.on_tool_complete(
    signature=glob_signature,
    result=[{"path": "/src/main.py"}],
)
# Automatically prefetches read_file for /src/main.py based on patterns

# When the predicted tool is called, result may already be cached
result = await prefetcher.await_prefetch(read_file_signature, timeout_ms=100)
if result:
    print("Prefetch hit - zero latency!")

# Check prefetch statistics
stats = prefetcher.stats
print(f"Prefetch hit rate: {stats.hit_rate:.1%}")
print(f"Latency saved: {stats.total_latency_saved_ms:.0f}ms")
```

### Schema Extraction

Extract and cache repeated JSON schemas for compression.

```python
from context_tools import SchemaExtractor, SchemaCache

extractor = SchemaExtractor(min_sample_size=5)
cache = SchemaCache(max_schemas=100)

# Extract schema from repeated objects
data = [
    {"id": 1, "name": "Alice", "active": True},
    {"id": 2, "name": "Bob", "active": False},
    # ... many similar objects
]

schema = extractor.extract(data, source_tool="get_users")
# Schema: {"id": "int", "name": "str", "active": "bool"}

# Compress using schema
compressed = extractor.compress_with_schema(data, schema)
# {"_schema_ref": "abc123", "_values": [[1, "Alice", True], [2, "Bob", False]]}

# Cache schema for reuse
schema_id = cache.put(schema)

# Decompress
original = extractor.decompress_with_schema(compressed, schema)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ToolPrefetcher                           │
│              (Predict → Execute → Cache)                     │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│ToolUsagePatterns│  │  ToolCallCache   │  │ ToolResultCompressor│
│               │  │                 │  │                     │
│ • Sequences   │  │ • Exact match   │  │ • Schema extraction │
│ • Predictions │  │ • Normalized    │  │ • List truncation   │
│ • Antipatterns│  │ • Semantic      │  │ • Field filtering   │
└───────────────┘  └─────────────────┘  └─────────────────────┘
```

## Cache Matching Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Exact** | Identical tool + args | Same call repeated |
| **Normalized** | Path normalization, whitespace | `/src/../src/main.py` = `/src/main.py` |
| **Semantic** | Embedding similarity | `read main.py` ≈ `read /src/main.py` |

## Compression Strategies

| Strategy | Compression | Reversible |
|----------|-------------|------------|
| Schema extraction | 2-4x | Yes |
| List truncation | 5-20x | Partial (stats preserved) |
| Low-value filtering | 1.5-2x | No |

## Tests

```bash
# Run all tests
uv run pytest packages/context-tools/tests -v

# Run with coverage
uv run pytest packages/context-tools/tests --cov=context_tools
```

**283 tests** covering:
- Cache with exact/normalized/semantic matching
- Usage pattern detection and prediction
- Antipattern detection
- Result compression with all strategies
- List truncation with 5 sampling strategies
- Schema extraction and caching
- Predictive prefetching with argument prediction
