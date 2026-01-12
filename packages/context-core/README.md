# context-core

> Foundation package for ContextEngine: Graph, Entities, Semantic Index, Token Budget

## Installation

```bash
uv pip install -e packages/context-core
```

## Components

### ContextGraph

Graph-based context representation with typed nodes and relationships.

```python
from context_core import ContextGraph, NodeType, EdgeType

# Create a graph
graph = ContextGraph(session_id="my-session")

# Add messages
msg = graph.add_message(role="user", content="Find all Python files")

# Add tool calls and results
call = graph.add_tool_call("glob", {"pattern": "**/*.py"})
result = graph.add_tool_result(call.id, ["main.py", "utils.py"])

# Query nodes
messages = graph.query_nodes(node_type=NodeType.MESSAGE)
recent = graph.query_nodes(min_importance=0.5)

# Get subgraph around a node
subgraph = graph.get_subgraph(call.id, depth=2)

# Serialize for LLM
messages = graph.to_messages()  # Returns list of message dicts
```

#### Node Types

| Type | Description |
|------|-------------|
| `MESSAGE` | User/assistant messages |
| `TOOL_CALL` | Tool invocations with arguments |
| `TOOL_RESULT` | Tool outputs |
| `ARTIFACT` | Generated files, code, data |
| `ENTITY` | Extracted named entities |
| `SUMMARY` | Compressed summaries |
| `SYSTEM` | System prompts and instructions |
| `MEMORY` | Retrieved memories |

#### Edge Types

| Type | Description |
|------|-------------|
| `TEMPORAL` | Time-ordered sequence |
| `CAUSAL` | Cause-effect relationship |
| `REFERENCES` | Content reference |
| `SUMMARIZES` | Summary of nodes |
| `TOOL_IO` | Tool call to result |
| `SAME_ENTITY` | Entity coreference |

### EntityTracker

NER-powered entity extraction with importance scoring.

```python
from context_core import EntityTracker

tracker = EntityTracker()

# Extract entities from text
entities = tracker.extract_from_text(
    "The user wants to refactor auth.py and update the User model"
)

# Get most important entities
important = tracker.get_most_important(top_k=10)

# Merge duplicate entities
tracker.merge_entities(entity1_id, entity2_id)

# Register custom patterns
tracker.register_pattern(
    name="file_path",
    pattern=r"[\w/]+\.(py|js|ts)",
    entity_type="FILE"
)
```

#### Built-in Patterns

- File paths (`*.py`, `src/main.js`)
- URLs (`https://...`)
- Python imports (`from x import y`)
- Class/function names (`class Foo`, `def bar`)
- Environment variables (`$HOME`, `${VAR}`)

### SemanticIndex

Embedding-based similarity search and duplicate detection.

```python
from context_core import SemanticIndex
from context_core.semantic import InMemoryVectorStore

# Create index with in-memory store
index = SemanticIndex(
    vector_store=InMemoryVectorStore()
)

# Index nodes from graph
index.index_nodes(graph.nodes)

# Search by semantic similarity
results = index.search(
    query="authentication flow",
    top_k=5,
    min_score=0.7
)

# Find near-duplicates
duplicates = index.find_duplicates(threshold=0.95)
```

#### Vector Store Backends

| Backend | Use Case |
|---------|----------|
| `InMemoryVectorStore` | Development, small datasets |
| `ChromaVectorStore` | Production, persistent storage |
| `QdrantVectorStore` | Production, scalable |

### TokenBudget

Token allocation with pre-rot threshold detection.

```python
from context_core import TokenBudget

# Create budget for 100K context window
budget = TokenBudget(
    total_tokens=100_000,
    warning_threshold=0.5,   # 50% - start monitoring
    trigger_threshold=0.65,  # 65% - trigger compression
)

# Allocate tokens to sections
budget.allocate("system", 2000)
budget.allocate("context", graph.total_tokens)
budget.allocate("response", 4000)

# Check status
status = budget.status
print(f"Usage: {status.usage_ratio:.1%}")
print(f"Needs compression: {status.needs_compression}")

# Get compression priority
priority = budget.get_compression_priority()
# Returns: {"context": 0.8, "system": 0.1, ...}
```

#### Budget Alerts

| Level | Threshold | Action |
|-------|-----------|--------|
| `INFO` | 50% | Monitor closely |
| `WARNING` | 65% | Trigger compression |
| `CRITICAL` | 85% | Aggressive compression |

### Tokenizer

Model-agnostic token counting with multiple backends.

```python
from context_core import get_tokenizer

# Auto-detect tokenizer
tokenizer = get_tokenizer("gpt-4")
count = tokenizer.count("Hello, world!")

# Specific implementations
from context_core.tokenizer import (
    TiktokenTokenizer,    # OpenAI models
    AnthropicTokenizer,   # Claude models
    EstimateTokenizer,    # Fallback (chars / 4)
)
```

## Data Models

All models are Pydantic-based with full validation:

```python
from context_core.graph import (
    ContextNode,
    NodeMetadata,
    Content,
    Edge,
    EdgeMetadata,
)

# Node with importance scoring
node = ContextNode(
    node_type=NodeType.MESSAGE,
    content=Content(text="Hello"),
    metadata=NodeMetadata(
        importance=0.8,
        access_count=5,
        pinned=False,
    )
)

# Automatic importance computation
importance = node.compute_importance()
```

## Testing

```bash
# Run all context-core tests
uv run pytest packages/context-core/tests/ -v

# Run specific test file
uv run pytest packages/context-core/tests/test_graph.py

# With coverage
uv run pytest packages/context-core/tests/ --cov=context_core
```

## API Reference

### ContextGraph

| Method | Description |
|--------|-------------|
| `add_message(role, content)` | Add a message node |
| `add_tool_call(name, arguments)` | Add a tool call node |
| `add_tool_result(call_id, result)` | Add a tool result node |
| `add_node(node)` | Add any node type |
| `get_node(id)` | Get node by ID |
| `remove_node(id)` | Remove a node |
| `query_nodes(**filters)` | Query with filters |
| `get_subgraph(id, depth)` | Extract subgraph |
| `to_messages()` | Convert to LLM format |
| `to_dict()` / `from_dict()` | Serialization |

### EntityTracker

| Method | Description |
|--------|-------------|
| `extract_from_text(text)` | Extract entities |
| `get_entity(id)` | Get entity by ID |
| `get_most_important(k)` | Top-k by importance |
| `merge_entities(id1, id2)` | Merge duplicates |
| `register_pattern(...)` | Add custom pattern |
| `to_dict()` / `from_dict()` | Serialization |

### SemanticIndex

| Method | Description |
|--------|-------------|
| `index_node(node)` | Index single node |
| `index_nodes(nodes)` | Batch index |
| `search(query, k)` | Similarity search |
| `find_duplicates(threshold)` | Find near-dupes |
| `remove_node(id)` | Remove from index |

### TokenBudget

| Method | Description |
|--------|-------------|
| `allocate(section, tokens)` | Allocate tokens |
| `deallocate(section)` | Free tokens |
| `status` | Get current status |
| `get_compression_priority()` | Section priorities |
| `summary()` | Human-readable summary |
