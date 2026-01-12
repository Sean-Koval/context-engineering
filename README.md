# ContextEngine

> An open-source context management platform for LLM-powered agents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-311%20passing-brightgreen.svg)]()

## The Problem

LLM agents experience **context degradation** well before hitting token limits. Quality drops significantly after ~50 tool calls, typically around 60-70% context capacity—a phenomenon we call **"pre-rot"**.

```
Quality │
   100% │████████████████████████
        │                        ████████
    80% │                                ████████
        │                                        ████
    60% │                                            ████
        │────────────────────────────────────────────────────
            0%    25%    50%    65%    80%    100%
                      Token Usage

        │◄──── Safe ────►│◄ Pre-Rot ►│◄── Degraded ──►│
```

## The Solution

ContextEngine treats context as a **first-class data structure** (graph-based, not flat strings) and applies intelligent compression through a **tiered strategy**:

1. **Lossless** (100% recoverable): Externalize payloads, deduplicate, collapse tool chains
2. **Compaction** (80-95% recoverable): Schema compression, entity-centric filtering
3. **Summarization** (last resort): Hierarchical, task-aware, and incremental summarization

## Features

- **Graph-based Context**: Typed nodes (messages, tool calls, artifacts) with relationships
- **Smart Compression**: 10-20x compression with minimal information loss
- **Pre-rot Detection**: Act before quality degrades, not after
- **Entity Tracking**: NER-powered entity extraction and importance scoring
- **Semantic Search**: Embedding-based similarity and duplicate detection
- **Recovery Manifests**: Track all compression operations for potential rollback
- **Observable**: OpenTelemetry tracing and Prometheus metrics built-in

## Installation

```bash
# Using uv (recommended)
uv add context-core context-compression

# Or with pip
pip install context-core context-compression
```

## Quick Start

```python
from context_core import ContextGraph, TokenBudget, EntityTracker, SemanticIndex
from context_compression import CompressionPipeline, CompressionTier

# Create a context graph
graph = ContextGraph(session_id="my-session")

# Add messages and tool calls
graph.add_message(role="user", content="Find all Python files in the src directory")
call = graph.add_tool_call("glob", {"pattern": "src/**/*.py"})
graph.add_tool_result(call.id, ["src/main.py", "src/utils.py", "src/config.py"])

# Track token budget with pre-rot detection
budget = TokenBudget(total_tokens=100_000)
budget.allocate("context", graph.total_tokens)

if budget.status.needs_compression:
    # Apply intelligent compression
    pipeline = CompressionPipeline()
    results = pipeline.compress(graph, max_tier=CompressionTier.COMPACTION)

    print(f"Saved {sum(r.tokens_saved for r in results)} tokens")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ContextEngine                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │context-core │  │  context-   │  │  context-   │              │
│  │             │  │ compression │  │   observe   │              │
│  │ • Graph     │  │             │  │             │              │
│  │ • Entities  │  │ • Pipeline  │  │ • Tracing   │              │
│  │ • Semantic  │  │ • Lossless  │  │ • Metrics   │              │
│  │ • Budget    │  │ • Compaction│  │ • Events    │              │
│  │ • Tokenizer │  │ • Summary   │  │             │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  context-   │  │  context-   │  │  context-   │  (Phase 3-4) │
│  │   memory    │  │    tools    │  │ multiagent  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Packages

| Package | Description | Status |
|---------|-------------|--------|
| `context-core` | Graph, entities, semantic index, token budget | ✅ Complete |
| `context-compression` | Compression pipeline with 9 strategies | ✅ Complete |
| `context-observe` | OpenTelemetry tracing, Prometheus metrics | ✅ Complete |
| `context-memory` | Storage backends, tiered storage, retrieval | 🚧 Phase 3 |
| `context-tools` | Tool caching, patterns, prefetching | 🚧 Phase 3 |
| `context-multiagent` | Broker, handoff, shared memory | 🚧 Phase 4 |

## Compression Strategies

### Lossless (100% Recoverable)

| Strategy | Description | Compression |
|----------|-------------|-------------|
| `ExternalizePayloads` | Move large outputs to external storage | 2-5x |
| `DeduplicateSemantically` | Remove near-duplicate content | 1.5-2x |
| `CollapseToolChains` | Merge sequential related tool calls | 2-3x |

### Compaction (80-95% Recoverable)

| Strategy | Description | Compression |
|----------|-------------|-------------|
| `SchemaCompression` | Extract and reference repeated JSON schemas | 2-4x |
| `EntityCentricCompression` | Keep only entity-relevant sentences | 2-3x |
| `TaskRelevanceCompression` | Filter by current task relevance | 2-4x |

### Summarization (Irreversible, Last Resort)

| Strategy | Description | Compression |
|----------|-------------|-------------|
| `HierarchicalSummarization` | Bottom-up multi-level summaries | 5-10x |
| `TaskAwareSummarization` | Task-focused with relevance scoring | 5-10x |
| `IncrementalSummarization` | Streaming updates to running summary | 5-10x |

## Development

```bash
# Clone the repository
git clone https://github.com/Sean-Koval/context-engineering.git
cd context-engineering

# Install dependencies
uv sync

# Install packages in development mode
uv pip install -e packages/context-core -e packages/context-compression -e packages/context-observe

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check --fix .

# Type check
uv run ty check .
```

## Project Structure

```
context-engineering/
├── packages/
│   ├── context-core/           # Foundation package
│   │   ├── src/context_core/
│   │   │   ├── graph/          # ContextGraph, nodes, edges
│   │   │   ├── entities/       # EntityTracker, NER backends
│   │   │   ├── semantic/       # SemanticIndex, vector stores
│   │   │   ├── budget/         # TokenBudget, pre-rot detection
│   │   │   └── tokenizer/      # Tokenizer protocol, implementations
│   │   └── tests/
│   ├── context-compression/    # Compression pipeline
│   │   ├── src/context_compression/
│   │   │   ├── strategies/
│   │   │   │   ├── lossless/   # Externalize, deduplicate, collapse
│   │   │   │   ├── compaction/ # Schema, entity-centric, task
│   │   │   │   └── summarization/ # Hierarchical, task-aware, incremental
│   │   │   ├── recovery/       # Manifest, operations
│   │   │   └── pipeline.py     # CompressionPipeline orchestrator
│   │   └── tests/
│   └── context-observe/        # Observability
│       ├── src/context_observe/
│       │   ├── tracer.py       # OpenTelemetry integration
│       │   ├── metrics.py      # Prometheus metrics
│       │   └── events.py       # Structured logging
│       └── tests/
├── specs/                      # Technical specifications
├── INDEX.md                    # Implementation progress tracking
├── TASK_BOARD.md              # Granular task breakdown
└── MASTER_ROADMAP.md          # Vision and architecture
```

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Foundation (Graph, Entities, Semantic, Budget) | ✅ Complete |
| **Phase 2** | Compression (Pipeline, 9 Strategies, Recovery) | ✅ Complete |
| **Phase 3** | Memory & Tools (Storage, Caching, Patterns) | 🚧 In Progress |
| **Phase 4** | Multi-Agent (Broker, Handoff, Sync) | 📅 Planned |

## Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Context utilization before degradation | 90%+ | ✅ |
| Reversible compression ratio | 3-5x | ✅ |
| Total compression ratio | 10-20x | ✅ |
| Test coverage | 90%+ | 311 tests |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### For Coding Agents

Each spec file in `specs/` contains:
- Complete Python code with type hints
- Pydantic models for all data structures
- Implementation checklists
- Test specifications

Use `TASK_BOARD.md` for granular task breakdown with dependencies.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [uv](https://github.com/astral-sh/uv) for blazing fast package management
- Uses [NetworkX](https://networkx.org/) for graph operations
- Embeddings powered by [sentence-transformers](https://www.sbert.net/)
- NER via [spaCy](https://spacy.io/)

---

**ContextEngine** - Because context is too valuable to waste.
