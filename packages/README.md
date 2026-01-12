# ContextEngine Packages

This directory contains the core packages that make up ContextEngine.

## Package Overview

```
packages/
├── context-core/        # Foundation: Graph, Entities, Semantic, Budget
├── context-compression/ # Compression Pipeline & Strategies
├── context-observe/     # Observability: Tracing, Metrics, Events
├── context-memory/      # (Phase 3) Storage & Retrieval
├── context-tools/       # (Phase 3) Tool Caching & Patterns
└── context-multiagent/  # (Phase 4) Multi-Agent Coordination
```

## Dependency Graph

```
                    ┌─────────────────┐
                    │  context-core   │
                    │   (foundation)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  context-   │  │  context-   │  │  context-   │
    │ compression │  │   observe   │  │   memory    │
    └──────┬──────┘  └─────────────┘  └──────┬──────┘
           │                                  │
           └──────────────┬───────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  context-tools  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    context-     │
                 │   multiagent    │
                 └─────────────────┘
```

## Package Status

| Package | Version | Status | Tests |
|---------|---------|--------|-------|
| `context-core` | 0.1.0 | ✅ Complete | 36 |
| `context-compression` | 0.1.0 | ✅ Complete | 275 |
| `context-observe` | 0.1.0 | ✅ Complete | ~20 |
| `context-memory` | - | 🚧 Phase 3 | - |
| `context-tools` | - | 🚧 Phase 3 | - |
| `context-multiagent` | - | 🚧 Phase 4 | - |

## Installation

### All Packages (Development)

```bash
# From repository root
uv sync
uv pip install -e packages/context-core \
               -e packages/context-compression \
               -e packages/context-observe
```

### Individual Packages

```bash
# Core only
uv pip install -e packages/context-core

# Core + Compression
uv pip install -e packages/context-core -e packages/context-compression

# With observability
uv pip install -e packages/context-observe
```

## Quick Reference

### context-core

```python
from context_core import (
    ContextGraph,      # Graph-based context representation
    EntityTracker,     # NER-powered entity extraction
    SemanticIndex,     # Embedding-based similarity search
    TokenBudget,       # Token allocation with pre-rot detection
    get_tokenizer,     # Model-specific token counting
)
```

### context-compression

```python
from context_compression import (
    CompressionPipeline,   # Orchestrates compression strategies
    CompressionTier,       # LOSSLESS, COMPACTION, SUMMARIZATION
    RecoveryManifest,      # Tracks operations for recovery
)

from context_compression.strategies import (
    # Lossless (100% recoverable)
    ExternalizePayloads,
    DeduplicateSemantically,
    CollapseToolChains,

    # Compaction (structure preserved)
    SchemaCompression,
    EntityCentricCompression,
    TaskRelevanceCompression,

    # Summarization (last resort)
    HierarchicalSummarization,
    TaskAwareSummarization,
    IncrementalSummarization,
)
```

### context-observe

```python
from context_observe import (
    ContextTracer,     # OpenTelemetry tracing
    ContextMetrics,    # Prometheus metrics
    ContextEvents,     # Structured logging
    traced,            # Decorator for automatic tracing
)
```

## Running Tests

```bash
# All packages
uv run pytest packages/

# Specific package
uv run pytest packages/context-core/tests/
uv run pytest packages/context-compression/tests/

# With coverage
uv run pytest packages/ --cov
```

## Creating a New Package

1. Create directory structure:
```bash
mkdir -p packages/context-{name}/src/context_{name}
mkdir -p packages/context-{name}/tests
```

2. Create `pyproject.toml`:
```toml
[project]
name = "context-{name}"
version = "0.1.0"
dependencies = ["context-core>=0.1.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

3. Create `__init__.py` with exports

4. Add tests in `tests/`

5. Install in development mode:
```bash
uv pip install -e packages/context-{name}
```
