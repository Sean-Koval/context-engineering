# Contributing to ContextEngine

Thank you for your interest in contributing to ContextEngine! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/Sean-Koval/context-engineering.git
cd context-engineering

# Install dependencies
uv sync

# Install all packages in development mode
uv pip install -e packages/context-core \
               -e packages/context-compression \
               -e packages/context-memory \
               -e packages/context-tools \
               -e packages/context-observe
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest packages/context-core/tests -v

# Run with coverage
uv run pytest --cov
```

### Code Quality

We use `ruff` for formatting and linting, and `ty` for type checking.

```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Type check
uv run ty check .
```

**Important**: Run these checks before committing. All PRs must pass CI checks.

## Code Style

### Type Hints

All functions must have type hints:

```python
from __future__ import annotations

def process_nodes(
    nodes: list[ContextNode],
    max_count: int = 100,
) -> list[ProcessedNode]:
    ...
```

### Pydantic Models

Use Pydantic for data models with validation:

```python
from pydantic import BaseModel, Field, field_validator

class TokenBudget(BaseModel):
    total_tokens: int = Field(gt=0, description="Total available tokens")
    warning_threshold: float = Field(default=0.5, ge=0, le=1)

    @field_validator("warning_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if v >= 1.0:
            raise ValueError("warning_threshold must be < 1.0")
        return v
```

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

### Naming Conventions

- Classes: `PascalCase` (e.g., `ContextGraph`, `TokenBudget`)
- Functions/methods: `snake_case` (e.g., `add_node`, `get_budget_status`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `DEFAULT_THRESHOLD`)
- Type aliases: `PascalCase` with `TypeAlias` annotation

## Project Structure

```
context-engineering/
├── packages/
│   ├── context-core/        # Graph, entities, semantic index, budget
│   ├── context-compression/ # Compression pipeline & strategies
│   ├── context-memory/      # Storage backends, tiered storage
│   ├── context-tools/       # Tool caching & patterns
│   ├── context-observe/     # Tracing, metrics, events
│   └── context-multiagent/  # Multi-agent coordination (Phase 4)
├── specs/                   # Technical specifications
├── docs/                    # Documentation and research
└── .github/workflows/       # CI/CD configuration
```

## Making Changes

### Branch Naming

- `feature/<description>` - New features
- `fix/<description>` - Bug fixes
- `docs/<description>` - Documentation updates
- `refactor/<description>` - Code refactoring

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(context-memory): Add Redis storage backend
fix(context-tools): Fix cache TTL expiration timing
docs: Update README with Phase 3 completion
```

### Pull Requests

1. Create a branch from `main`
2. Make your changes
3. Ensure all tests pass: `uv run pytest`
4. Ensure code quality: `uv run ruff format . && uv run ruff check --fix .`
5. Update documentation if needed
6. Submit PR with clear description

### PR Description Template

```markdown
## Summary
Brief description of changes

## Changes
- List of specific changes
- Another change

## Test Plan
- [ ] Unit tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Related Issues
Fixes #123
```

## For Coding Agents

Each spec file in `specs/` contains:
- Complete Python code with type hints
- Pydantic models for all data structures
- Implementation checklists
- Test specifications

Use `TASK_BOARD.md` for granular task breakdown with dependencies.

### Key Files

| File | Purpose |
|------|---------|
| `INDEX.md` | Master checklist - implementation progress |
| `MASTER_ROADMAP.md` | Vision, architecture, phases |
| `API_CONTRACTS.md` | Package interface definitions |
| `specs/*.md` | Detailed technical specifications |

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- Use discussions for questions and ideas

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
