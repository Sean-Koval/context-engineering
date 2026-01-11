# CLAUDE.md - Project Intelligence

> This file is read by Claude Code at the start of every session.
> Keep it updated with learnings, patterns, and project-specific knowledge.
> The team should contribute to this file multiple times per week.

## Project Overview

**ContextEngine** is an open-source context management platform for LLM-powered agents. It addresses context degradation that occurs after ~50 tool calls, where quality drops significantly before hitting token limits (around 60-70% capacity - "pre-rot").

The key insight: Context should be treated as a first-class data structure (graph-based), not a flat string. Compression should prioritize reversible compaction over irreversible summarization.

## Tech Stack

- **Python 3.12+** with full type hints
- **uv** for package management
- **ruff** for linting & formatting
- **ty** for type checking
- **Pydantic 2.0+** for data models
- **networkx** for graph operations
- **sentence-transformers** for embeddings
- **tiktoken** for token counting
- **spacy** for NER
- **chromadb** for vector storage
- **OpenTelemetry** for observability
- **structlog** for structured logging

## Project Structure

```
context-engine/
├── packages/
│   ├── context-core/        # Graph, entities, semantic index, token budget
│   ├── context-compression/ # Compression pipeline & strategies
│   ├── context-memory/      # Storage backends, tiered storage
│   ├── context-tools/       # Tool caching & patterns
│   ├── context-observe/     # Tracing, metrics, events
│   └── context-multiagent/  # Broker, handoff, sync
├── specs/                   # Technical specifications
├── INDEX.md                 # Master index with checklists
├── MASTER_ROADMAP.md        # Vision, architecture, phases
└── API_CONTRACTS.md         # Package interfaces
```

## Implementation Phases

### Phase 1: Foundation (Current)
- context-core (ContextGraph, EntityTracker, SemanticIndex, TokenBudget, Tokenizer)
- Basic context-observe (tracing, metrics)

### Phase 2: Compression
- context-compression (pipeline, 8 strategies, recovery)
- Basic context-memory (MemoryStore protocol, FileSystemStore)

### Phase 3: Memory & Tools
- Full context-memory (Postgres, Redis, tiered storage)
- context-tools (cache, patterns, compression, prefetch)

### Phase 4: Integration & Multi-Agent
- context-multiagent (broker, handoff, sync)
- Framework integrations (LangChain, LlamaIndex)

## Code Style & Conventions

### Naming Conventions
- Classes: `PascalCase` (e.g., `ContextGraph`, `TokenBudget`)
- Functions/methods: `snake_case` (e.g., `add_node`, `get_budget_status`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- Type aliases: `PascalCase` with `TypeAlias` annotation

### File Organization
- One class per file for major components
- Group related utilities in `utils.py`
- Models in `models.py` or dedicated module
- Tests mirror source structure in `tests/`

### Import Order
```python
from __future__ import annotations

# Standard library
import json
from typing import TypeAlias

# Third-party
import networkx as nx
from pydantic import BaseModel

# Local
from context_core.models import ContextNode
```

## Key Data Structures

### Node Types
`MESSAGE`, `TOOL_CALL`, `TOOL_RESULT`, `ARTIFACT`, `ENTITY`, `SUMMARY`, `SYSTEM`, `MEMORY`

### Edge Types
`TEMPORAL`, `CAUSAL`, `REFERENCES`, `SUMMARIZES`, `CONTRADICTS`, `DEPENDS_ON`, `SAME_ENTITY`, `PARENT_CHILD`, `TOOL_IO`

### Compression Levels
- Level 0: `FULL` (original)
- Level 1: `COMPACTED` (reversible - externalized, deduplicated, collapsed)
- Level 2: `SUMMARIZED` (irreversible but logged)
- Level 3: `EVICTED` (pointer only)

## Common Patterns

### Pattern: Pydantic Model with Validation
```python
from pydantic import BaseModel, Field, field_validator

class TokenBudget(BaseModel):
    total_tokens: int = Field(gt=0, description="Total available tokens")
    warning_threshold: float = Field(default=0.5, ge=0, le=1)
    trigger_threshold: float = Field(default=0.65, ge=0, le=1)

    @field_validator("trigger_threshold")
    @classmethod
    def trigger_after_warning(cls, v: float, info) -> float:
        if v <= info.data.get("warning_threshold", 0.5):
            raise ValueError("trigger_threshold must exceed warning_threshold")
        return v
```

### Pattern: Protocol for Pluggable Backends
```python
from typing import Protocol

class MemoryStore(Protocol):
    async def save(self, key: str, data: bytes) -> None: ...
    async def load(self, key: str) -> bytes | None: ...
    async def delete(self, key: str) -> bool: ...
```

## Things Claude Should NOT Do

- Do NOT use `any` type - always define proper types
- Do NOT use pip directly - use `uv add` for packages
- Do NOT skip type hints on function signatures
- Do NOT use mutable default arguments (use `Field(default_factory=list)`)
- Do NOT forget to add `from __future__ import annotations` for forward refs
- Do NOT implement features outside the current phase scope
- Do NOT create new files without checking if similar functionality exists
- Do NOT use naive truncation for context - use the compression pipeline

## Python Development Rules

> These rules apply to Python projects using the uv/ruff/ty toolchain.

### Package Management
- ALWAYS use `uv` for package management, NEVER use pip directly
- Add dependencies with `uv add package-name`
- Add dev dependencies with `uv add --dev package-name`
- Run commands in the venv with `uv run command`

### Code Quality
- Run `uv run ruff format .` before committing to format code
- Run `uv run ruff check --fix .` to lint and auto-fix issues
- Run `uv run ty check .` to verify type correctness

### Type Hints
- Use type hints on ALL function signatures
- Use `from __future__ import annotations` for forward references
- Prefer Pydantic models for data validation
- Use `TypeAlias` for complex type definitions

### Testing
- Run tests with `uv run pytest`
- Write tests in `tests/` directory
- Use pytest fixtures for setup/teardown

## Testing Requirements

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_specific.py

# Run tests for a specific package
uv run pytest packages/context-core/tests/
```

## Build & Deploy

```bash
# Install dependencies
uv sync

# Run development server
uv run python -m context_engine

# Build package
uv build

# Lint & format
uv run ruff format . && uv run ruff check --fix .

# Type check
uv run ty check .
```

## Key Files to Know About

| File | Purpose |
|------|---------|
| `INDEX.md` | Master checklist - START HERE for implementation |
| `MASTER_ROADMAP.md` | Vision, architecture, 24-week plan |
| `API_CONTRACTS.md` | Package interface definitions |
| `TASK_BOARD.md` | Current task tracking |
| `specs/01-context-core.md` | Foundation spec (graph, entities, semantic, budget) |
| `specs/02-context-compression.md` | Compression pipeline spec |
| `specs/03-context-memory.md` | Storage backends spec |
| `specs/04-context-tools.md` | Tool optimization spec |
| `specs/05-context-observe.md` | Observability spec |
| `specs/06-context-multiagent.md` | Multi-agent coordination spec |

## MCP Tools Available

### serena (LSP Code Intelligence)
- Use for finding symbol definitions
- Use for finding all usages/references
- Use for safe refactoring (rename symbols)
- Activate project first: "Activate project at /path/to/project"

### code-mode (UTCP Dynamic Tools)
- Register external APIs as callable tools
- Chain API calls with TypeScript
- Search for tools by description

## Multi-Agent Worktree Management

Use git worktrees to run multiple Claude Code sessions in parallel without conflicts.

### Why Worktrees?
- **Isolation**: Each agent gets its own working directory
- **Parallel work**: Multiple agents can build/test independently
- **Clean coordination**: Merge through normal git flow
- **No context staleness**: Files match agent's mental model

### Worktree CLI Tool

Located at `scripts/worktree/wt.sh` - manages worktrees with consistent naming.

```bash
# List all worktrees
./scripts/worktree/wt.sh list

# Create an agent session (auto-numbered)
./scripts/worktree/wt.sh create agent
# Creates: ../context-engineering-agent-session-001

# Create a feature worktree
./scripts/worktree/wt.sh create feature compression-pipeline --package context-compression

# Check worktree status
./scripts/worktree/wt.sh status feature compression-pipeline

# Sync with main
./scripts/worktree/wt.sh sync feature compression-pipeline

# Finish and create PR
./scripts/worktree/wt.sh finish feature compression-pipeline

# Remove worktree
./scripts/worktree/wt.sh remove feature compression-pipeline
```

### Worktree Types

| Type | Purpose | Naming |
|------|---------|--------|
| `agent` | Auto-numbered sessions | `agent/session-001` |
| `feature` | New features | `feature/<name>` |
| `fix` | Bug fixes | `fix/<issue>` |
| `experiment` | Experimental work | `experiment/<name>` |
| `review` | Code review | `review/<pr-num>` |

### Naming Conventions

**Directory**: `{repo}-{type}-{identifier}`
```
~/repos/
├── context-engineering/                    # main worktree
├── context-engineering-agent-session-001/  # agent session
├── context-engineering-feature-compression/ # feature work
└── context-engineering-fix-token-overflow/ # bug fix
```

**Branch**: `{type}/{identifier}`
```
main
agent/session-001
feature/compression-pipeline
fix/token-overflow
```

### Agent Session Workflow

1. **Start session**: `./scripts/worktree/wt.sh create agent`
2. **Work in worktree**: `cd ../context-engineering-agent-session-001 && claude`
3. **Sync periodically**: `./scripts/worktree/wt.sh sync agent session-001`
4. **Complete work**: `./scripts/worktree/wt.sh finish agent session-001 --pr`
5. **Cleanup**: `./scripts/worktree/wt.sh remove agent session-001`

### Session Registry

Active sessions tracked in `.worktree-sessions.json` (gitignored).
View with: `./scripts/worktree/wt.sh sessions`

### Best Practices for Multi-Agent Work

- **One package per agent**: Assign each agent to a specific package
- **Sync before finish**: Always sync with main before creating PR
- **Clear task scope**: Define clear boundaries in TASK_BOARD.md
- **Commit often**: Small, focused commits for easier merges
- **Use PR reviews**: Let agents review each other's PRs

## Success Metrics

| Metric | Target |
|--------|--------|
| Context utilization before degradation | 90%+ |
| Reversible compression ratio | 3-5x |
| Total compression ratio (with summarization) | 10-20x |
| Memory retrieval p99 latency | < 100ms |
| Tool cache hit rate | > 60% |
| Multi-agent transfer overhead | < 5% |

## Current Team Focus

Phase 1: Foundation - Building context-core package
- ContextGraph with CRUD operations
- EntityTracker with NER
- SemanticIndex with embeddings
- TokenBudget with pre-rot detection

## Verification Steps

Before considering a task complete, Claude should:

1. **Format**: `uv run ruff format .`
2. **Lint**: `uv run ruff check --fix .`
3. **Type check**: `uv run ty check .`
4. **Test**: `uv run pytest`

---
*Last updated: 2026-01-10 by Claude*
