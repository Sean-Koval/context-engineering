# Context-Compression: Detailed Implementation Specification

> **Phase 2 Compression** | Estimated: 4-6 weeks | Priority: P0
> This document provides implementable specifications for the compression pipeline.

---

## Table of Contents
1. [Package Overview](#package-overview)
2. [Compression Philosophy](#compression-philosophy)
3. [Component 1: CompressionPipeline](#component-1-compressionpipeline)
4. [Component 2: Lossless Strategies](#component-2-lossless-strategies)
5. [Component 3: Compaction Strategies](#component-3-compaction-strategies)
6. [Component 4: Summarization Strategies](#component-4-summarization-strategies)
7. [Component 5: RecoveryManifest](#component-5-recoverymanifest)
8. [Integration with context-core](#integration-with-context-core)
9. [Task Breakdown](#task-breakdown)
10. [Test Specifications](#test-specifications)

---

## Package Overview

### Purpose
`context-compression` provides a multi-strategy compression pipeline that intelligently reduces context size while preserving information recovery capability.

### Core Principle: Compression Hierarchy
```
┌─────────────────────────────────────────────────────────────────────┐
│                    COMPRESSION HIERARCHY                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Level 0: FULL                                                       │
│  ├── Original content, no modifications                              │
│  └── Token cost: 100%                                                │
│                                                                      │
│  Level 1: LOSSLESS (Always try first)                               │
│  ├── ExternalizePayloads: Move large outputs to storage             │
│  ├── DeduplicateSemantically: Remove near-duplicate content          │
│  ├── CollapseToolChains: Merge sequential tool calls                 │
│  └── Token cost: 30-50% | Recovery: 100%                            │
│                                                                      │
│  Level 2: COMPACTED (Reversible with context loss)                  │
│  ├── SchemaCompression: Extract and reference schemas                │
│  ├── EntityCentricCompression: Compress around entities              │
│  ├── TaskRelevanceCompression: Remove off-task content               │
│  └── Token cost: 10-30% | Recovery: 80-95%                          │
│                                                                      │
│  Level 3: SUMMARIZED (Irreversible, last resort)                    │
│  ├── HierarchicalSummarization: Multi-level summaries                │
│  ├── TaskAwareSummarization: Task-focused summaries                  │
│  ├── IncrementalSummarization: Streaming summaries                   │
│  └── Token cost: 5-10% | Recovery: ~60%                             │
│                                                                      │
│  Level 4: EVICTED                                                    │
│  ├── Content removed, pointer only                                   │
│  └── Token cost: <1% | Recovery: via manifest                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Dependencies
```toml
[project]
name = "context-compression"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "context-core>=0.1.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
llm = ["anthropic>=0.18", "openai>=1.0"]
all = ["context-compression[llm]"]
```

### Module Structure
```
context_compression/
├── __init__.py
├── py.typed
├── pipeline.py              # CompressionPipeline orchestrator
├── types.py                 # Shared types, CompressionResult
├── strategies/
│   ├── __init__.py
│   ├── base.py              # CompressionStrategy protocol
│   ├── lossless/
│   │   ├── __init__.py
│   │   ├── externalize.py   # ExternalizePayloads
│   │   ├── deduplicate.py   # DeduplicateSemantically
│   │   └── collapse.py      # CollapseToolChains
│   ├── compaction/
│   │   ├── __init__.py
│   │   ├── schema.py        # SchemaCompression
│   │   ├── entity.py        # EntityCentricCompression
│   │   └── task.py          # TaskRelevanceCompression
│   └── summarization/
│       ├── __init__.py
│       ├── hierarchical.py  # HierarchicalSummarization
│       ├── task_aware.py    # TaskAwareSummarization
│       └── incremental.py   # IncrementalSummarization
├── recovery/
│   ├── __init__.py
│   ├── manifest.py          # RecoveryManifest
│   └── operations.py        # Operation types
└── preservers/
    ├── __init__.py
    └── rules.py             # PreservationRule definitions
```

---

## Compression Philosophy

### Why Compaction Before Summarization?

| Approach | Information Loss | Recoverability | When to Use |
|----------|------------------|----------------|-------------|
| **Lossless** | 0% | 100% | Always first |
| **Compaction** | 5-20% | 80-95% | Before summarization |
| **Summarization** | 40-60% | ~60% | Last resort |

### The Pre-Rot Problem

```
Token Usage vs. Quality Graph:

Quality │
   100% │████████████████████████
        │                        ████████
    80% │                                ████████
        │                                        ████
    60% │                                            ████
        │                                                ███
    40% │                                                   ██
        │────────────────────────────────────────────────────────
            0%    25%    50%    65%    80%    100%
                      Token Usage

        │◄──── Safe ────►│◄ Pre-Rot ►│◄── Degraded ──►│

Key Insight: Quality degrades BEFORE hitting token limits.
Action: Compress proactively at 50-65%, not reactively at 95%+.
```

### Preservation Rules

Some content should NEVER be compressed:
1. **Recent context**: Last N messages (configurable, default 10)
2. **Pinned nodes**: Explicitly marked as important
3. **Active entities**: Entities referenced in recent messages
4. **System prompts**: Core instructions
5. **Error traces**: Recent errors for debugging

---

## Component 1: CompressionPipeline

### 1.1 Type Definitions

```python
# context_compression/types.py
from enum import Enum, auto
from typing import Any, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from context_core.graph.types import CompressionLevel

class CompressionTier(str, Enum):
    """Tier of compression strategy."""
    LOSSLESS = "lossless"
    COMPACTION = "compaction"
    SUMMARIZATION = "summarization"


class CompressionResult(BaseModel):
    """Result of a compression operation."""
    success: bool
    strategy_name: str
    tier: CompressionTier

    # Token metrics
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int

    # Node metrics
    nodes_processed: int
    nodes_compressed: int
    nodes_removed: int
    nodes_created: int  # For summaries

    # Timing
    duration_ms: float

    # Recovery
    is_recoverable: bool
    manifest_id: Optional[UUID] = None

    @property
    def compression_ratio(self) -> float:
        """Ratio of original to compressed (higher = more compression)."""
        if self.compressed_tokens == 0:
            return float('inf')
        return self.original_tokens / self.compressed_tokens

    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved."""
        if self.original_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.original_tokens) * 100


class CompressionPlan(BaseModel):
    """A plan for compression before execution."""
    strategies: list[str]
    target_tokens: Optional[int] = None
    estimated_savings: int
    nodes_affected: list[UUID]
    preservations: list[str]  # Reasons nodes are preserved


class PreservationRule(BaseModel):
    """Rule for preserving nodes from compression."""
    name: str
    description: str
    priority: int = 0  # Higher = more important to preserve

    # Criteria (any match = preserve)
    node_types: Optional[list[str]] = None
    min_importance: Optional[float] = None
    max_age_seconds: Optional[int] = None
    required_tags: Optional[set[str]] = None
    entity_ids: Optional[list[str]] = None
    pinned: bool = False
```

### 1.2 Strategy Protocol

```python
# context_compression/strategies/base.py
from typing import Protocol, Optional, runtime_checkable
from context_core.graph import ContextGraph
from ..types import CompressionResult, CompressionTier
from ..recovery.manifest import RecoveryManifest

@runtime_checkable
class CompressionStrategy(Protocol):
    """Protocol for compression strategies."""

    @property
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    def tier(self) -> CompressionTier:
        """Compression tier (lossless, compaction, summarization)."""
        ...

    @property
    def priority(self) -> int:
        """Execution priority within tier (lower = run first)."""
        ...

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        """Estimate tokens that would be saved."""
        ...

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """
        Execute compression on the graph.

        Args:
            graph: The context graph to compress (modified in place)
            manifest: Recovery manifest to log operations
            target_node_ids: Specific nodes to target (None = all eligible)
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics
        """
        ...

    def can_apply(self, graph: ContextGraph) -> bool:
        """Check if this strategy can be applied to the graph."""
        ...
```

### 1.3 CompressionPipeline Class

```python
# context_compression/pipeline.py
from typing import Optional, Callable
from uuid import UUID
import time
from datetime import datetime, timedelta

from context_core.graph import ContextGraph, ContextNode, NodeType
from context_core.budget import TokenBudget

from .types import (
    CompressionResult, CompressionPlan, CompressionTier,
    PreservationRule
)
from .strategies.base import CompressionStrategy
from .recovery.manifest import RecoveryManifest


class CompressionPipeline:
    """
    Orchestrates multi-strategy compression.

    Features:
    - Strategy ordering by tier and priority
    - Preservation rules
    - Target-based compression (compress until target reached)
    - Dry-run planning
    - Recovery manifest management
    """

    DEFAULT_PRESERVATION_RULES = [
        PreservationRule(
            name="recent_messages",
            description="Preserve recent messages for context continuity",
            priority=100,
            max_age_seconds=300,  # Last 5 minutes
        ),
        PreservationRule(
            name="pinned_content",
            description="Never compress pinned nodes",
            priority=100,
            pinned=True,
        ),
        PreservationRule(
            name="system_prompts",
            description="Preserve system configuration",
            priority=90,
            node_types=["system"],
        ),
        PreservationRule(
            name="high_importance",
            description="Preserve high-importance nodes",
            priority=80,
            min_importance=0.9,
        ),
    ]

    def __init__(
        self,
        strategies: Optional[list[CompressionStrategy]] = None,
        preservation_rules: Optional[list[PreservationRule]] = None,
        preserve_recent_n: int = 10,
        enable_recovery: bool = True,
        on_compression: Optional[Callable[[CompressionResult], None]] = None,
    ):
        self._strategies: list[CompressionStrategy] = strategies or []
        self._preservation_rules = preservation_rules or self.DEFAULT_PRESERVATION_RULES.copy()
        self._preserve_recent_n = preserve_recent_n
        self._enable_recovery = enable_recovery
        self._on_compression = on_compression

        # Sort strategies by tier then priority
        self._sort_strategies()

    def _sort_strategies(self):
        """Sort strategies: lossless first, then compaction, then summarization."""
        tier_order = {
            CompressionTier.LOSSLESS: 0,
            CompressionTier.COMPACTION: 1,
            CompressionTier.SUMMARIZATION: 2,
        }
        self._strategies.sort(
            key=lambda s: (tier_order.get(s.tier, 99), s.priority)
        )

    def register_strategy(self, strategy: CompressionStrategy):
        """Register a new compression strategy."""
        self._strategies.append(strategy)
        self._sort_strategies()

    def add_preservation_rule(self, rule: PreservationRule):
        """Add a preservation rule."""
        self._preservation_rules.append(rule)

    def get_preserved_nodes(self, graph: ContextGraph) -> set[UUID]:
        """Get IDs of nodes that should be preserved."""
        preserved = set()
        now = datetime.utcnow()

        # Always preserve most recent N nodes
        recent = graph.get_recent(self._preserve_recent_n)
        for node in recent:
            preserved.add(node.id)

        # Apply preservation rules
        for node in graph:
            for rule in self._preservation_rules:
                if self._matches_rule(node, rule, now):
                    preserved.add(node.id)
                    break

        return preserved

    def _matches_rule(
        self,
        node: ContextNode,
        rule: PreservationRule,
        now: datetime,
    ) -> bool:
        """Check if a node matches a preservation rule."""
        if rule.pinned and node.metadata.pinned:
            return True

        if rule.node_types and node.type.value not in rule.node_types:
            return False

        if rule.min_importance is not None:
            if node.compute_importance() < rule.min_importance:
                return False

        if rule.max_age_seconds is not None:
            age = (now - node.metadata.created_at).total_seconds()
            if age > rule.max_age_seconds:
                return False

        if rule.required_tags:
            if not rule.required_tags & node.metadata.tags:
                return False

        if rule.entity_ids:
            if not any(e in node.metadata.entities for e in rule.entity_ids):
                return False

        return True

    def plan(
        self,
        graph: ContextGraph,
        target_tokens: Optional[int] = None,
        max_tier: CompressionTier = CompressionTier.SUMMARIZATION,
    ) -> CompressionPlan:
        """
        Create a compression plan without executing.

        Args:
            graph: The context graph to analyze
            target_tokens: Target token count to achieve
            max_tier: Maximum compression tier to include

        Returns:
            CompressionPlan with estimated results
        """
        preserved = self.get_preserved_nodes(graph)
        compressible = [n.id for n in graph if n.id not in preserved]

        tier_order = {
            CompressionTier.LOSSLESS: 0,
            CompressionTier.COMPACTION: 1,
            CompressionTier.SUMMARIZATION: 2,
        }
        max_tier_value = tier_order.get(max_tier, 2)

        strategies_to_use = []
        estimated_total = 0

        for strategy in self._strategies:
            if tier_order.get(strategy.tier, 99) > max_tier_value:
                continue

            if not strategy.can_apply(graph):
                continue

            estimated = strategy.estimate_savings(graph, compressible)
            if estimated > 0:
                strategies_to_use.append(strategy.name)
                estimated_total += estimated

                if target_tokens and estimated_total >= target_tokens:
                    break

        preservation_reasons = []
        for rule in self._preservation_rules:
            matching = sum(
                1 for n in graph
                if self._matches_rule(n, rule, datetime.utcnow())
            )
            if matching > 0:
                preservation_reasons.append(f"{rule.name}: {matching} nodes")

        return CompressionPlan(
            strategies=strategies_to_use,
            target_tokens=target_tokens,
            estimated_savings=estimated_total,
            nodes_affected=compressible,
            preservations=preservation_reasons,
        )

    def compress(
        self,
        graph: ContextGraph,
        target_tokens: Optional[int] = None,
        max_tier: CompressionTier = CompressionTier.SUMMARIZATION,
        dry_run: bool = False,
    ) -> list[CompressionResult]:
        """
        Execute compression pipeline.

        Args:
            graph: Context graph to compress (modified in place)
            target_tokens: Stop when this many tokens saved
            max_tier: Maximum compression tier to use
            dry_run: If True, estimate only without modifying

        Returns:
            List of CompressionResult for each strategy applied
        """
        if dry_run:
            plan = self.plan(graph, target_tokens, max_tier)
            # Return estimated results
            return [
                CompressionResult(
                    success=True,
                    strategy_name="DRY_RUN",
                    tier=CompressionTier.LOSSLESS,
                    original_tokens=0,
                    compressed_tokens=0,
                    tokens_saved=plan.estimated_savings,
                    nodes_processed=len(plan.nodes_affected),
                    nodes_compressed=0,
                    nodes_removed=0,
                    nodes_created=0,
                    duration_ms=0,
                    is_recoverable=True,
                )
            ]

        # Create recovery manifest
        manifest = RecoveryManifest(
            session_id=graph.session_id,
            enable_recovery=self._enable_recovery,
        )

        preserved = self.get_preserved_nodes(graph)
        compressible = [n.id for n in graph if n.id not in preserved]

        tier_order = {
            CompressionTier.LOSSLESS: 0,
            CompressionTier.COMPACTION: 1,
            CompressionTier.SUMMARIZATION: 2,
        }
        max_tier_value = tier_order.get(max_tier, 2)

        results = []
        total_saved = 0

        for strategy in self._strategies:
            # Check tier limit
            if tier_order.get(strategy.tier, 99) > max_tier_value:
                continue

            # Check if strategy can apply
            if not strategy.can_apply(graph):
                continue

            # Check if we've reached target
            if target_tokens and total_saved >= target_tokens:
                break

            # Calculate remaining target
            remaining_target = None
            if target_tokens:
                remaining_target = target_tokens - total_saved

            # Execute strategy
            start = time.perf_counter()
            try:
                result = strategy.compress(
                    graph=graph,
                    manifest=manifest,
                    target_node_ids=compressible,
                    target_tokens=remaining_target,
                )
                result.duration_ms = (time.perf_counter() - start) * 1000
                result.manifest_id = manifest.id

                results.append(result)
                total_saved += result.tokens_saved

                if self._on_compression:
                    self._on_compression(result)

                # Update compressible list (some nodes may have been removed)
                compressible = [
                    nid for nid in compressible
                    if nid in graph
                ]

            except Exception as e:
                results.append(CompressionResult(
                    success=False,
                    strategy_name=strategy.name,
                    tier=strategy.tier,
                    original_tokens=0,
                    compressed_tokens=0,
                    tokens_saved=0,
                    nodes_processed=0,
                    nodes_compressed=0,
                    nodes_removed=0,
                    nodes_created=0,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    is_recoverable=False,
                ))

        return results

    def compress_to_budget(
        self,
        graph: ContextGraph,
        budget: TokenBudget,
        aggressive: bool = False,
    ) -> list[CompressionResult]:
        """
        Compress graph to fit within token budget.

        Args:
            graph: Context graph to compress
            budget: Token budget with current allocation
            aggressive: If True, use summarization tier

        Returns:
            List of compression results
        """
        # Calculate how many tokens we need to free
        excess = budget.used_tokens - budget.available_tokens
        if excess <= 0:
            return []  # Already within budget

        # Add buffer (compress a bit more than needed)
        target = int(excess * 1.2)

        max_tier = (
            CompressionTier.SUMMARIZATION if aggressive
            else CompressionTier.COMPACTION
        )

        return self.compress(
            graph=graph,
            target_tokens=target,
            max_tier=max_tier,
        )

    def get_manifest(self, manifest_id: UUID) -> Optional[RecoveryManifest]:
        """Retrieve a recovery manifest by ID."""
        # In production, this would query persistent storage
        # For now, manifests are ephemeral
        return None
```

---

## Component 2: Lossless Strategies

### 2.1 ExternalizePayloads

Moves large content to external storage, replacing with reference.

```python
# context_compression/strategies/lossless/externalize.py
from typing import Optional, Protocol
from uuid import UUID, uuid4
import json
import hashlib

from context_core.graph import ContextGraph, ContextNode, NodeType
from context_core.graph.types import CompressionLevel

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import ExternalizeOperation
from ..base import CompressionStrategy


class ExternalStorage(Protocol):
    """Protocol for external payload storage."""

    def store(self, key: str, data: bytes) -> str:
        """Store data and return URI."""
        ...

    def retrieve(self, uri: str) -> bytes:
        """Retrieve data by URI."""
        ...


class InMemoryExternalStorage:
    """Simple in-memory storage for testing."""

    def __init__(self):
        self._store: dict[str, bytes] = {}

    def store(self, key: str, data: bytes) -> str:
        uri = f"memory://{key}"
        self._store[uri] = data
        return uri

    def retrieve(self, uri: str) -> bytes:
        return self._store.get(uri, b"")


class ExternalizePayloads(CompressionStrategy):
    """
    Move large payloads to external storage.

    Targets:
    - Large tool results (>threshold tokens)
    - Artifact content
    - Verbose outputs

    Preserves:
    - Reference to original content
    - Schema/structure summary
    - First N lines as preview
    """

    def __init__(
        self,
        storage: ExternalStorage,
        min_tokens: int = 500,          # Only externalize above this
        preview_tokens: int = 100,       # Keep this many tokens as preview
        tokenizer: Optional["Tokenizer"] = None,
    ):
        self._storage = storage
        self._min_tokens = min_tokens
        self._preview_tokens = preview_tokens
        self._tokenizer = tokenizer

    @property
    def name(self) -> str:
        return "externalize_payloads"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def priority(self) -> int:
        return 10  # Run early

    def can_apply(self, graph: ContextGraph) -> bool:
        """Check if there are any nodes worth externalizing."""
        for node in graph:
            if self._is_candidate(node):
                return True
        return False

    def _is_candidate(self, node: ContextNode) -> bool:
        """Check if node is a candidate for externalization."""
        if node.compression_level >= CompressionLevel.COMPACTED:
            return False  # Already compressed

        if node.content.external_ref:
            return False  # Already externalized

        # Check token count
        tokens = node.token_count or self._estimate_tokens(node)
        if tokens < self._min_tokens:
            return False

        # Target types
        if node.type in (NodeType.TOOL_RESULT, NodeType.ARTIFACT):
            return True

        return False

    def _estimate_tokens(self, node: ContextNode) -> int:
        """Estimate token count for a node."""
        text = self._get_content_text(node)
        if self._tokenizer:
            return self._tokenizer.count_tokens(text)
        return len(text) // 4  # Rough estimate

    def _get_content_text(self, node: ContextNode) -> str:
        """Extract text content from node."""
        if node.content.text:
            return node.content.text
        if node.content.tool_output:
            return json.dumps(node.content.tool_output, default=str)
        if node.content.artifact_data:
            return str(node.content.artifact_data)
        return ""

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        """Estimate tokens that would be saved."""
        savings = 0
        for node in graph:
            if target_node_ids and node.id not in target_node_ids:
                continue
            if self._is_candidate(node):
                tokens = node.token_count or self._estimate_tokens(node)
                # We save tokens minus preview
                savings += max(0, tokens - self._preview_tokens - 50)  # 50 for reference
        return savings

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute externalization."""
        original_tokens = 0
        compressed_tokens = 0
        nodes_processed = 0
        nodes_compressed = 0
        tokens_saved = 0

        candidates = [
            n for n in graph
            if self._is_candidate(n)
            and (not target_node_ids or n.id in target_node_ids)
        ]

        # Sort by size (largest first for maximum impact)
        candidates.sort(
            key=lambda n: n.token_count or self._estimate_tokens(n),
            reverse=True,
        )

        for node in candidates:
            if target_tokens and tokens_saved >= target_tokens:
                break

            nodes_processed += 1
            content_text = self._get_content_text(node)
            content_tokens = node.token_count or self._estimate_tokens(node)
            original_tokens += content_tokens

            # Create preview
            preview = self._create_preview(content_text, node)

            # Store externally
            content_hash = hashlib.sha256(content_text.encode()).hexdigest()[:16]
            key = f"{graph.session_id}/{node.id}/{content_hash}"
            uri = self._storage.store(key, content_text.encode())

            # Log operation for recovery
            manifest.log_operation(ExternalizeOperation(
                node_id=node.id,
                external_uri=uri,
                original_tokens=content_tokens,
                original_content_hash=content_hash,
            ))

            # Update node
            node.content.original_tokens = content_tokens
            node.content.external_ref = uri

            # Replace content with preview
            if node.content.tool_output:
                node.content.tool_output = {
                    "_externalized": True,
                    "_uri": uri,
                    "_preview": preview,
                    "_original_tokens": content_tokens,
                }
            elif node.content.artifact_data:
                node.content.artifact_data = {
                    "_externalized": True,
                    "_uri": uri,
                    "_preview": preview,
                    "_original_tokens": content_tokens,
                }
            elif node.content.text:
                node.content.text = f"[Externalized: {content_tokens} tokens]\n{preview}\n[Full content: {uri}]"

            # Update compression level
            node.compression_level = CompressionLevel.COMPACTED

            # Calculate new token count
            new_tokens = self._estimate_tokens(node)
            node.token_count = new_tokens
            node.content.compressed_tokens = new_tokens
            compressed_tokens += new_tokens
            tokens_saved += content_tokens - new_tokens
            nodes_compressed += 1

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=0,
            is_recoverable=True,
        )

    def _create_preview(self, content: str, node: ContextNode) -> str:
        """Create a preview of the content."""
        lines = content.split('\n')

        # For structured data, try to extract schema
        if node.type == NodeType.TOOL_RESULT:
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return f"Keys: {list(data.keys())[:10]}"
                elif isinstance(data, list):
                    return f"List of {len(data)} items"
            except:
                pass

        # Default: first few lines
        preview_lines = []
        tokens = 0
        for line in lines[:20]:
            line_tokens = len(line) // 4
            if tokens + line_tokens > self._preview_tokens:
                break
            preview_lines.append(line)
            tokens += line_tokens

        return '\n'.join(preview_lines) + ('\n...' if len(lines) > len(preview_lines) else '')
```

### 2.2 DeduplicateSemantically

Removes near-duplicate content, keeping references.

```python
# context_compression/strategies/lossless/deduplicate.py
from typing import Optional
from uuid import UUID

from context_core.graph import ContextGraph, ContextNode, NodeType, EdgeType
from context_core.graph.types import CompressionLevel
from context_core.semantic import SemanticIndex

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import DeduplicateOperation
from ..base import CompressionStrategy


class DeduplicateSemantically(CompressionStrategy):
    """
    Remove semantically duplicate nodes.

    Targets:
    - Repeated tool outputs
    - Similar message content
    - Redundant information

    Preserves:
    - Most recent/important version
    - Reference edges to canonical node
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        similarity_threshold: float = 0.92,
        min_tokens_to_dedupe: int = 50,
    ):
        self._semantic_index = semantic_index
        self._similarity_threshold = similarity_threshold
        self._min_tokens = min_tokens_to_dedupe

    @property
    def name(self) -> str:
        return "deduplicate_semantically"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def priority(self) -> int:
        return 20

    def can_apply(self, graph: ContextGraph) -> bool:
        return len(graph) >= 2

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        """Estimate by finding duplicate pairs."""
        duplicates = self._semantic_index.find_duplicates(
            threshold=self._similarity_threshold
        )
        savings = 0
        seen = set()

        for id1, id2, score in duplicates:
            if target_node_ids:
                if id1 not in target_node_ids and id2 not in target_node_ids:
                    continue

            if id2 in seen:
                continue
            seen.add(id2)

            node = graph.get_node(id2)
            if node and node.token_count:
                savings += node.token_count

        return savings

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute deduplication."""
        duplicates = self._semantic_index.find_duplicates(
            threshold=self._similarity_threshold
        )

        original_tokens = 0
        tokens_saved = 0
        nodes_removed = 0
        nodes_processed = 0

        # Group duplicates, keeping highest importance as canonical
        duplicate_groups: dict[UUID, list[tuple[UUID, float]]] = {}

        for id1, id2, score in duplicates:
            if target_node_ids:
                if id1 not in target_node_ids and id2 not in target_node_ids:
                    continue

            node1 = graph.get_node(id1)
            node2 = graph.get_node(id2)

            if not node1 or not node2:
                continue

            # Determine canonical (higher importance or more recent)
            if node1.compute_importance() >= node2.compute_importance():
                canonical, duplicate = id1, id2
            else:
                canonical, duplicate = id2, id1

            if canonical not in duplicate_groups:
                duplicate_groups[canonical] = []
            duplicate_groups[canonical].append((duplicate, score))

        # Remove duplicates
        removed = set()
        for canonical_id, dupes in duplicate_groups.items():
            for dupe_id, score in dupes:
                if target_tokens and tokens_saved >= target_tokens:
                    break

                if dupe_id in removed:
                    continue

                dupe_node = graph.get_node(dupe_id)
                if not dupe_node:
                    continue

                nodes_processed += 1
                dupe_tokens = dupe_node.token_count or 0
                original_tokens += dupe_tokens

                # Log operation
                manifest.log_operation(DeduplicateOperation(
                    removed_node_id=dupe_id,
                    canonical_node_id=canonical_id,
                    similarity_score=score,
                    removed_tokens=dupe_tokens,
                ))

                # Add reference edge before removal
                from context_core.graph.edges import Edge
                graph.add_edge(Edge(
                    source_id=canonical_id,
                    target_id=dupe_id,  # Will be removed but logged
                    type=EdgeType.REFERENCES,
                ))

                # Remove duplicate
                graph.remove_node(dupe_id)
                self._semantic_index.remove_node(dupe_id)
                removed.add(dupe_id)
                nodes_removed += 1
                tokens_saved += dupe_tokens

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=0,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=0,
            nodes_removed=nodes_removed,
            nodes_created=0,
            duration_ms=0,
            is_recoverable=True,
        )
```

### 2.3 CollapseToolChains

Merges sequential related tool calls into single summary nodes.

```python
# context_compression/strategies/lossless/collapse.py
from typing import Optional
from uuid import UUID
from collections import defaultdict

from context_core.graph import ContextGraph, ContextNode, NodeType, EdgeType
from context_core.graph.types import CompressionLevel
from context_core.graph.nodes import Content, NodeMetadata

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import CollapseOperation
from ..base import CompressionStrategy


class CollapseToolChains(CompressionStrategy):
    """
    Collapse sequential tool call chains into summaries.

    Targets:
    - Read operations on same file
    - Sequential search refinements
    - Directory traversals
    - API pagination

    Preserves:
    - Final result
    - Chain summary
    - Recovery information
    """

    def __init__(
        self,
        min_chain_length: int = 3,
        same_tool_only: bool = False,
    ):
        self._min_chain_length = min_chain_length
        self._same_tool_only = same_tool_only

    @property
    def name(self) -> str:
        return "collapse_tool_chains"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def priority(self) -> int:
        return 30

    def can_apply(self, graph: ContextGraph) -> bool:
        # Need at least min_chain_length tool calls
        tool_calls = graph.query_nodes(node_types=[NodeType.TOOL_CALL])
        return len(tool_calls) >= self._min_chain_length

    def _find_chains(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> list[list[ContextNode]]:
        """Find collapsible tool chains."""
        chains = []
        current_chain = []

        tool_nodes = [
            n for n in graph.query_nodes(
                node_types=[NodeType.TOOL_CALL, NodeType.TOOL_RESULT],
                order_by="sequence",
            )
            if not target_node_ids or n.id in target_node_ids
        ]

        for node in tool_nodes:
            if node.type == NodeType.TOOL_CALL:
                if current_chain and self._can_extend_chain(current_chain, node):
                    current_chain.append(node)
                else:
                    if len(current_chain) >= self._min_chain_length * 2:  # call+result pairs
                        chains.append(current_chain)
                    current_chain = [node]
            elif node.type == NodeType.TOOL_RESULT:
                if current_chain:
                    current_chain.append(node)

        if len(current_chain) >= self._min_chain_length * 2:
            chains.append(current_chain)

        return chains

    def _can_extend_chain(
        self,
        chain: list[ContextNode],
        node: ContextNode,
    ) -> bool:
        """Check if node can extend the current chain."""
        if not chain:
            return True

        # Find last tool call in chain
        last_call = None
        for n in reversed(chain):
            if n.type == NodeType.TOOL_CALL:
                last_call = n
                break

        if not last_call:
            return True

        # Check if same tool (if required)
        if self._same_tool_only:
            if last_call.content.tool_name != node.content.tool_name:
                return False

        # Check if related tools
        related_tools = {
            'read_file': {'read_file', 'glob', 'grep'},
            'glob': {'read_file', 'glob', 'grep'},
            'grep': {'read_file', 'glob', 'grep'},
            'list_dir': {'list_dir', 'read_file'},
            'search': {'search', 'fetch'},
        }

        last_tool = last_call.content.tool_name
        current_tool = node.content.tool_name

        if last_tool in related_tools:
            return current_tool in related_tools[last_tool]

        return last_tool == current_tool

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        chains = self._find_chains(graph, target_node_ids)
        savings = 0

        for chain in chains:
            chain_tokens = sum(n.token_count or 0 for n in chain)
            # Estimate collapsed size as ~20% of original
            collapsed_tokens = int(chain_tokens * 0.2)
            savings += chain_tokens - collapsed_tokens

        return savings

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute chain collapse."""
        chains = self._find_chains(graph, target_node_ids)

        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_removed = 0
        nodes_created = 0
        nodes_processed = 0

        for chain in chains:
            if target_tokens and tokens_saved >= target_tokens:
                break

            chain_tokens = sum(n.token_count or 0 for n in chain)
            original_tokens += chain_tokens
            nodes_processed += len(chain)

            # Create summary node
            summary = self._create_chain_summary(chain, graph)
            summary_tokens = summary.token_count or 0
            compressed_tokens += summary_tokens

            # Log operation
            manifest.log_operation(CollapseOperation(
                chain_node_ids=[n.id for n in chain],
                summary_node_id=summary.id,
                original_tokens=chain_tokens,
                compressed_tokens=summary_tokens,
                chain_length=len(chain),
            ))

            # Add summary to graph
            graph.add_node(summary, connect_temporal=False)
            nodes_created += 1

            # Remove chain nodes (except add SUMMARIZES edges first)
            for node in chain:
                from context_core.graph.edges import Edge
                graph.add_edge(Edge(
                    source_id=summary.id,
                    target_id=node.id,
                    type=EdgeType.SUMMARIZES,
                ))

            for node in chain:
                graph.remove_node(node.id)
                nodes_removed += 1

            tokens_saved += chain_tokens - summary_tokens

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=0,
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=0,
            is_recoverable=True,
        )

    def _create_chain_summary(
        self,
        chain: list[ContextNode],
        graph: ContextGraph,
    ) -> ContextNode:
        """Create a summary node for a tool chain."""
        # Extract tool names and patterns
        tool_calls = [n for n in chain if n.type == NodeType.TOOL_CALL]
        tool_results = [n for n in chain if n.type == NodeType.TOOL_RESULT]

        tool_names = [n.content.tool_name for n in tool_calls]
        tool_counts = defaultdict(int)
        for name in tool_names:
            tool_counts[name] += 1

        # Get last result as the "outcome"
        last_result = tool_results[-1] if tool_results else None
        outcome_preview = ""
        if last_result and last_result.content.tool_output:
            outcome = last_result.content.tool_output
            if isinstance(outcome, str):
                outcome_preview = outcome[:200]
            elif isinstance(outcome, dict):
                outcome_preview = f"Result with keys: {list(outcome.keys())[:5]}"
            elif isinstance(outcome, list):
                outcome_preview = f"List of {len(outcome)} items"

        # Build summary text
        summary_parts = [
            f"[Tool Chain Summary: {len(chain)} operations]",
            f"Tools used: {dict(tool_counts)}",
        ]

        if outcome_preview:
            summary_parts.append(f"Final result: {outcome_preview}")

        summary_text = "\n".join(summary_parts)

        return ContextNode(
            type=NodeType.SUMMARY,
            content=Content(
                text=summary_text,
                summarized_node_ids=[n.id for n in chain],
                summary_method="collapse_tool_chains",
            ),
            metadata=NodeMetadata(
                importance=max(n.metadata.importance for n in chain),
                tags={"tool_chain_summary"},
            ),
            compression_level=CompressionLevel.COMPACTED,
            token_count=len(summary_text) // 4,
        )
```

---

## Component 3: Compaction Strategies

### 3.1 SchemaCompression

Extracts and references repeated schemas.

```python
# context_compression/strategies/compaction/schema.py
from typing import Optional, Any
from uuid import UUID
import json
import hashlib
from collections import defaultdict

from context_core.graph import ContextGraph, ContextNode, NodeType
from context_core.graph.types import CompressionLevel

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import SchemaOperation
from ..base import CompressionStrategy


class SchemaCompression(CompressionStrategy):
    """
    Extract and deduplicate JSON schemas from structured data.

    Targets:
    - Repeated API responses with same structure
    - Tool results with consistent schemas
    - Lists of objects with shared structure

    Preserves:
    - Schema definition (extracted once)
    - Data values with schema reference
    """

    def __init__(
        self,
        min_occurrences: int = 3,
        min_schema_tokens: int = 100,
    ):
        self._min_occurrences = min_occurrences
        self._min_schema_tokens = min_schema_tokens
        self._extracted_schemas: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return "schema_compression"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def priority(self) -> int:
        return 10

    def can_apply(self, graph: ContextGraph) -> bool:
        return len(graph.query_nodes(node_types=[NodeType.TOOL_RESULT])) >= self._min_occurrences

    def _extract_schema(self, data: Any, max_depth: int = 5) -> dict:
        """Extract JSON schema from data."""
        if max_depth <= 0:
            return {"type": "any"}

        if data is None:
            return {"type": "null"}
        elif isinstance(data, bool):
            return {"type": "boolean"}
        elif isinstance(data, int):
            return {"type": "integer"}
        elif isinstance(data, float):
            return {"type": "number"}
        elif isinstance(data, str):
            return {"type": "string"}
        elif isinstance(data, list):
            if len(data) == 0:
                return {"type": "array", "items": {"type": "any"}}
            # Sample first few items for schema
            item_schemas = [self._extract_schema(item, max_depth - 1) for item in data[:3]]
            # Use most common schema
            return {"type": "array", "items": item_schemas[0]}
        elif isinstance(data, dict):
            properties = {}
            for key, value in list(data.items())[:20]:  # Limit keys
                properties[key] = self._extract_schema(value, max_depth - 1)
            return {"type": "object", "properties": properties}
        else:
            return {"type": "unknown"}

    def _schema_hash(self, schema: dict) -> str:
        """Create hash of schema for deduplication."""
        return hashlib.md5(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]

    def _find_schema_occurrences(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> dict[str, list[ContextNode]]:
        """Find nodes with same schema."""
        schema_nodes: dict[str, list[ContextNode]] = defaultdict(list)

        for node in graph.query_nodes(node_types=[NodeType.TOOL_RESULT]):
            if target_node_ids and node.id not in target_node_ids:
                continue

            if node.compression_level >= CompressionLevel.COMPACTED:
                continue

            data = node.content.tool_output
            if not isinstance(data, (dict, list)):
                continue

            schema = self._extract_schema(data)
            schema_key = self._schema_hash(schema)
            schema_nodes[schema_key].append(node)

            if schema_key not in self._extracted_schemas:
                self._extracted_schemas[schema_key] = schema

        return schema_nodes

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        occurrences = self._find_schema_occurrences(graph, target_node_ids)
        savings = 0

        for schema_key, nodes in occurrences.items():
            if len(nodes) < self._min_occurrences:
                continue

            schema = self._extracted_schemas[schema_key]
            schema_tokens = len(json.dumps(schema)) // 4

            if schema_tokens < self._min_schema_tokens:
                continue

            # Savings: schema extracted once, each node saves ~30%
            for node in nodes[1:]:  # First node keeps full schema
                node_tokens = node.token_count or 0
                savings += int(node_tokens * 0.3)

        return savings

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute schema compression."""
        occurrences = self._find_schema_occurrences(graph, target_node_ids)

        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_compressed = 0
        nodes_processed = 0

        for schema_key, nodes in occurrences.items():
            if len(nodes) < self._min_occurrences:
                continue

            if target_tokens and tokens_saved >= target_tokens:
                break

            schema = self._extracted_schemas[schema_key]
            schema_tokens = len(json.dumps(schema)) // 4

            if schema_tokens < self._min_schema_tokens:
                continue

            # First node becomes schema definition
            schema_node = nodes[0]

            # Compress remaining nodes
            for node in nodes[1:]:
                if target_tokens and tokens_saved >= target_tokens:
                    break

                nodes_processed += 1
                node_tokens = node.token_count or 0
                original_tokens += node_tokens

                # Create compressed representation
                compressed_data = {
                    "_schema_ref": schema_key,
                    "_values": self._extract_values(node.content.tool_output, schema),
                }

                manifest.log_operation(SchemaOperation(
                    node_id=node.id,
                    schema_key=schema_key,
                    original_tokens=node_tokens,
                ))

                # Update node
                node.content.tool_output = compressed_data
                node.compression_level = CompressionLevel.COMPACTED

                new_tokens = len(json.dumps(compressed_data)) // 4
                node.token_count = new_tokens
                compressed_tokens += new_tokens
                tokens_saved += node_tokens - new_tokens
                nodes_compressed += 1

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=0,
            is_recoverable=True,
        )

    def _extract_values(self, data: Any, schema: dict) -> Any:
        """Extract just values, stripping redundant structure."""
        if schema.get("type") == "object":
            if isinstance(data, dict):
                # Just keep values, keys are in schema
                return list(data.values())
        elif schema.get("type") == "array":
            if isinstance(data, list):
                item_schema = schema.get("items", {})
                return [self._extract_values(item, item_schema) for item in data]
        return data
```

### 3.2 EntityCentricCompression

Compresses around entity mentions, keeping entity context.

```python
# context_compression/strategies/compaction/entity.py
from typing import Optional
from uuid import UUID
from collections import defaultdict

from context_core.graph import ContextGraph, ContextNode, NodeType
from context_core.graph.types import CompressionLevel
from context_core.entities import EntityTracker

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import EntityCompressOperation
from ..base import CompressionStrategy


class EntityCentricCompression(CompressionStrategy):
    """
    Compress content while preserving entity context.

    Strategy:
    1. Identify important entities in conversation
    2. For each node, keep sentences mentioning important entities
    3. Summarize/remove sentences without entity mentions

    Preserves:
    - All mentions of tracked entities
    - Context around entity mentions
    - Entity relationships
    """

    def __init__(
        self,
        entity_tracker: EntityTracker,
        context_sentences: int = 1,  # Sentences before/after entity mention
        min_node_tokens: int = 100,
    ):
        self._entity_tracker = entity_tracker
        self._context_sentences = context_sentences
        self._min_node_tokens = min_node_tokens

    @property
    def name(self) -> str:
        return "entity_centric_compression"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def priority(self) -> int:
        return 20

    def can_apply(self, graph: ContextGraph) -> bool:
        # Need entities to compress around
        important_entities = self._entity_tracker.get_most_important(5)
        return len(important_entities) > 0

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_has_entity(
        self,
        sentence: str,
        entity_names: set[str],
    ) -> bool:
        """Check if sentence mentions any tracked entity."""
        sentence_lower = sentence.lower()
        for name in entity_names:
            if name.lower() in sentence_lower:
                return True
        return False

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        important_entities = self._entity_tracker.get_most_important(10)
        entity_names = set()
        for e in important_entities:
            entity_names.add(e.canonical_name)
            entity_names.update(e.aliases)

        savings = 0

        for node in graph.query_nodes(node_types=[NodeType.MESSAGE]):
            if target_node_ids and node.id not in target_node_ids:
                continue

            if node.compression_level >= CompressionLevel.COMPACTED:
                continue

            if not node.content.text:
                continue

            node_tokens = node.token_count or len(node.content.text) // 4
            if node_tokens < self._min_node_tokens:
                continue

            sentences = self._split_sentences(node.content.text)
            entity_sentences = sum(
                1 for s in sentences
                if self._sentence_has_entity(s, entity_names)
            )

            # Estimate: keep entity sentences + context, remove rest
            if len(sentences) > entity_sentences * (1 + 2 * self._context_sentences):
                removable = len(sentences) - entity_sentences * (1 + 2 * self._context_sentences)
                savings += int(node_tokens * (removable / len(sentences)) * 0.8)

        return savings

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute entity-centric compression."""
        important_entities = self._entity_tracker.get_most_important(10)
        entity_names = set()
        for e in important_entities:
            entity_names.add(e.canonical_name)
            entity_names.update(e.aliases)

        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_compressed = 0
        nodes_processed = 0

        for node in graph.query_nodes(node_types=[NodeType.MESSAGE]):
            if target_node_ids and node.id not in target_node_ids:
                continue

            if target_tokens and tokens_saved >= target_tokens:
                break

            if node.compression_level >= CompressionLevel.COMPACTED:
                continue

            if not node.content.text:
                continue

            node_tokens = node.token_count or len(node.content.text) // 4
            if node_tokens < self._min_node_tokens:
                continue

            nodes_processed += 1
            original_tokens += node_tokens

            sentences = self._split_sentences(node.content.text)

            # Find sentences with entities
            keep_indices = set()
            for i, sentence in enumerate(sentences):
                if self._sentence_has_entity(sentence, entity_names):
                    # Keep this sentence and context
                    for j in range(
                        max(0, i - self._context_sentences),
                        min(len(sentences), i + self._context_sentences + 1)
                    ):
                        keep_indices.add(j)

            if len(keep_indices) >= len(sentences) - 1:
                # Not enough to compress
                compressed_tokens += node_tokens
                continue

            # Build compressed text
            compressed_parts = []
            removed_count = 0

            for i, sentence in enumerate(sentences):
                if i in keep_indices:
                    compressed_parts.append(sentence)
                else:
                    removed_count += 1

            if removed_count > 0:
                compressed_parts.append(f"[{removed_count} sentences omitted]")

            compressed_text = " ".join(compressed_parts)

            manifest.log_operation(EntityCompressOperation(
                node_id=node.id,
                original_sentences=len(sentences),
                kept_sentences=len(keep_indices),
                removed_sentences=removed_count,
                entities_preserved=[e.canonical_name for e in important_entities],
            ))

            # Update node
            original_text = node.content.text
            node.content.text = compressed_text
            node.compression_level = CompressionLevel.COMPACTED

            new_tokens = len(compressed_text) // 4
            node.token_count = new_tokens
            node.content.original_tokens = node_tokens
            node.content.compressed_tokens = new_tokens

            compressed_tokens += new_tokens
            tokens_saved += node_tokens - new_tokens
            nodes_compressed += 1

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=0,
            is_recoverable=True,
        )
```

---

## Component 4: Summarization Strategies

### 4.1 HierarchicalSummarization

Creates multi-level summaries of context.

```python
# context_compression/strategies/summarization/hierarchical.py
from typing import Optional, Protocol
from uuid import UUID
from datetime import datetime, timedelta

from context_core.graph import ContextGraph, ContextNode, NodeType, EdgeType
from context_core.graph.types import CompressionLevel
from context_core.graph.nodes import Content, NodeMetadata

from ...types import CompressionResult, CompressionTier
from ...recovery.manifest import RecoveryManifest
from ...recovery.operations import SummarizeOperation
from ..base import CompressionStrategy


class LLMSummarizer(Protocol):
    """Protocol for LLM-based summarization."""

    def summarize(
        self,
        texts: list[str],
        max_tokens: int,
        instruction: Optional[str] = None,
    ) -> str:
        """Summarize texts into max_tokens."""
        ...


class HierarchicalSummarization(CompressionStrategy):
    """
    Create hierarchical summaries of context.

    Levels:
    1. Message-level: Summarize long individual messages
    2. Chunk-level: Summarize groups of messages
    3. Session-level: High-level session summary

    Preserves:
    - Key decisions and actions
    - Entity mentions
    - Task progress
    """

    def __init__(
        self,
        summarizer: LLMSummarizer,
        chunk_size: int = 10,           # Messages per chunk
        message_threshold: int = 500,    # Tokens to trigger message-level
        chunk_threshold: int = 2000,     # Tokens to trigger chunk-level
    ):
        self._summarizer = summarizer
        self._chunk_size = chunk_size
        self._message_threshold = message_threshold
        self._chunk_threshold = chunk_threshold

    @property
    def name(self) -> str:
        return "hierarchical_summarization"

    @property
    def tier(self) -> CompressionTier:
        return CompressionTier.SUMMARIZATION

    @property
    def priority(self) -> int:
        return 10

    def can_apply(self, graph: ContextGraph) -> bool:
        messages = graph.query_nodes(node_types=[NodeType.MESSAGE])
        return len(messages) >= self._chunk_size

    def estimate_savings(
        self,
        graph: ContextGraph,
        target_node_ids: Optional[list[UUID]] = None,
    ) -> int:
        nodes = graph.query_nodes(
            node_types=[NodeType.MESSAGE],
            max_compression_level=CompressionLevel.COMPACTED,
        )

        if target_node_ids:
            nodes = [n for n in nodes if n.id in target_node_ids]

        total_tokens = sum(n.token_count or 0 for n in nodes)

        # Estimate: summarization typically achieves 5-10x compression
        return int(total_tokens * 0.8)  # 80% savings estimate

    def compress(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: Optional[list[UUID]] = None,
        target_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """Execute hierarchical summarization."""
        nodes = graph.query_nodes(
            node_types=[NodeType.MESSAGE],
            max_compression_level=CompressionLevel.COMPACTED,
            order_by="sequence",
        )

        if target_node_ids:
            nodes = [n for n in nodes if n.id in target_node_ids]

        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_removed = 0
        nodes_created = 0
        nodes_processed = 0

        # Group into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0

        for node in nodes:
            node_tokens = node.token_count or 0
            current_chunk.append(node)
            current_tokens += node_tokens

            if len(current_chunk) >= self._chunk_size or current_tokens >= self._chunk_threshold:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

        if current_chunk:
            chunks.append(current_chunk)

        # Summarize each chunk
        for chunk in chunks:
            if target_tokens and tokens_saved >= target_tokens:
                break

            chunk_tokens = sum(n.token_count or 0 for n in chunk)
            original_tokens += chunk_tokens
            nodes_processed += len(chunk)

            # Build chunk text
            chunk_texts = []
            for node in chunk:
                role = node.content.role.value if node.content.role else "unknown"
                text = node.content.text or ""
                chunk_texts.append(f"{role}: {text}")

            # Generate summary
            summary_text = self._summarizer.summarize(
                texts=chunk_texts,
                max_tokens=min(200, chunk_tokens // 5),
                instruction="Summarize this conversation chunk, preserving key decisions, actions, and entity mentions.",
            )

            # Create summary node
            summary_node = ContextNode(
                type=NodeType.SUMMARY,
                content=Content(
                    text=summary_text,
                    summarized_node_ids=[n.id for n in chunk],
                    summary_method="hierarchical",
                ),
                metadata=NodeMetadata(
                    importance=max(n.metadata.importance for n in chunk),
                    tags={"hierarchical_summary"},
                    created_at=chunk[0].metadata.created_at,
                ),
                compression_level=CompressionLevel.SUMMARIZED,
                token_count=len(summary_text) // 4,
            )

            # Log operation
            manifest.log_operation(SummarizeOperation(
                original_node_ids=[n.id for n in chunk],
                summary_node_id=summary_node.id,
                original_tokens=chunk_tokens,
                summary_tokens=summary_node.token_count,
                method="hierarchical",
            ))

            # Add summary and create edges
            graph.add_node(summary_node, connect_temporal=False)
            nodes_created += 1

            for node in chunk:
                from context_core.graph.edges import Edge
                graph.add_edge(Edge(
                    source_id=summary_node.id,
                    target_id=node.id,
                    type=EdgeType.SUMMARIZES,
                ))

            # Remove original nodes
            for node in chunk:
                graph.remove_node(node.id)
                nodes_removed += 1

            compressed_tokens += summary_node.token_count or 0
            tokens_saved += chunk_tokens - (summary_node.token_count or 0)

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=0,
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=0,
            is_recoverable=False,  # Summarization is not fully recoverable
        )
```

---

## Component 5: RecoveryManifest

### 5.1 Operation Types

```python
# context_compression/recovery/operations.py
from typing import Any, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class OperationType(str, Enum):
    EXTERNALIZE = "externalize"
    DEDUPLICATE = "deduplicate"
    COLLAPSE = "collapse"
    SCHEMA = "schema"
    ENTITY_COMPRESS = "entity_compress"
    SUMMARIZE = "summarize"


class CompressionOperation(BaseModel):
    """Base class for compression operations."""
    id: UUID = Field(default_factory=lambda: uuid4())
    type: OperationType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    node_id: Optional[UUID] = None


class ExternalizeOperation(CompressionOperation):
    type: OperationType = OperationType.EXTERNALIZE
    external_uri: str
    original_tokens: int
    original_content_hash: str


class DeduplicateOperation(CompressionOperation):
    type: OperationType = OperationType.DEDUPLICATE
    removed_node_id: UUID
    canonical_node_id: UUID
    similarity_score: float
    removed_tokens: int


class CollapseOperation(CompressionOperation):
    type: OperationType = OperationType.COLLAPSE
    chain_node_ids: list[UUID]
    summary_node_id: UUID
    original_tokens: int
    compressed_tokens: int
    chain_length: int


class SchemaOperation(CompressionOperation):
    type: OperationType = OperationType.SCHEMA
    schema_key: str
    original_tokens: int


class EntityCompressOperation(CompressionOperation):
    type: OperationType = OperationType.ENTITY_COMPRESS
    original_sentences: int
    kept_sentences: int
    removed_sentences: int
    entities_preserved: list[str]


class SummarizeOperation(CompressionOperation):
    type: OperationType = OperationType.SUMMARIZE
    original_node_ids: list[UUID]
    summary_node_id: UUID
    original_tokens: int
    summary_tokens: int
    method: str
```

### 5.2 RecoveryManifest

```python
# context_compression/recovery/manifest.py
from typing import Optional, Union
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field

from .operations import (
    CompressionOperation, OperationType,
    ExternalizeOperation, DeduplicateOperation, CollapseOperation,
    SchemaOperation, EntityCompressOperation, SummarizeOperation,
)


class ManifestStats(BaseModel):
    """Statistics about the manifest."""
    total_operations: int
    operations_by_type: dict[str, int]
    total_tokens_saved: int
    total_nodes_affected: int
    recoverable_operations: int
    non_recoverable_operations: int


class RecoveryManifest:
    """
    Tracks compression operations for potential recovery.

    Features:
    - Operation logging
    - Node recovery (for lossless/compaction operations)
    - Audit trail
    - Statistics
    """

    def __init__(
        self,
        session_id: str,
        enable_recovery: bool = True,
    ):
        self.id = uuid4()
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.enable_recovery = enable_recovery

        self._operations: list[CompressionOperation] = []
        self._node_operations: dict[UUID, list[CompressionOperation]] = {}

    def log_operation(
        self,
        operation: CompressionOperation,
    ) -> None:
        """Log a compression operation."""
        self._operations.append(operation)

        if operation.node_id:
            if operation.node_id not in self._node_operations:
                self._node_operations[operation.node_id] = []
            self._node_operations[operation.node_id].append(operation)

    def get_node_operations(
        self,
        node_id: UUID,
    ) -> list[CompressionOperation]:
        """Get all operations affecting a node."""
        return self._node_operations.get(node_id, [])

    def get_operations_by_type(
        self,
        operation_type: OperationType,
    ) -> list[CompressionOperation]:
        """Get operations of a specific type."""
        return [op for op in self._operations if op.type == operation_type]

    def can_recover_node(self, node_id: UUID) -> bool:
        """Check if a node can be fully recovered."""
        operations = self.get_node_operations(node_id)

        for op in operations:
            # Summarization is not recoverable
            if op.type == OperationType.SUMMARIZE:
                return False

        return True

    def get_recovery_instructions(
        self,
        node_id: UUID,
    ) -> list[dict]:
        """Get instructions for recovering a node."""
        operations = self.get_node_operations(node_id)
        instructions = []

        for op in reversed(operations):  # Reverse order for recovery
            if op.type == OperationType.EXTERNALIZE:
                instructions.append({
                    "action": "fetch_external",
                    "uri": op.external_uri,
                    "expected_hash": op.original_content_hash,
                })
            elif op.type == OperationType.DEDUPLICATE:
                instructions.append({
                    "action": "restore_from_canonical",
                    "canonical_id": str(op.canonical_node_id),
                })
            elif op.type == OperationType.SCHEMA:
                instructions.append({
                    "action": "expand_schema",
                    "schema_key": op.schema_key,
                })
            elif op.type == OperationType.ENTITY_COMPRESS:
                instructions.append({
                    "action": "manual_review",
                    "note": "Entity compression removed non-entity sentences",
                })

        return instructions

    def stats(self) -> ManifestStats:
        """Get manifest statistics."""
        ops_by_type: dict[str, int] = {}
        tokens_saved = 0
        nodes_affected = set()
        recoverable = 0
        non_recoverable = 0

        for op in self._operations:
            type_name = op.type.value
            ops_by_type[type_name] = ops_by_type.get(type_name, 0) + 1

            if op.node_id:
                nodes_affected.add(op.node_id)

            # Calculate tokens saved
            if hasattr(op, 'original_tokens') and hasattr(op, 'compressed_tokens'):
                tokens_saved += op.original_tokens - op.compressed_tokens
            elif hasattr(op, 'removed_tokens'):
                tokens_saved += op.removed_tokens
            elif hasattr(op, 'original_tokens') and hasattr(op, 'summary_tokens'):
                tokens_saved += op.original_tokens - op.summary_tokens

            # Track recoverability
            if op.type == OperationType.SUMMARIZE:
                non_recoverable += 1
            else:
                recoverable += 1

        return ManifestStats(
            total_operations=len(self._operations),
            operations_by_type=ops_by_type,
            total_tokens_saved=tokens_saved,
            total_nodes_affected=len(nodes_affected),
            recoverable_operations=recoverable,
            non_recoverable_operations=non_recoverable,
        )

    def to_dict(self) -> dict:
        """Serialize manifest."""
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "operations": [op.model_dump(mode="json") for op in self._operations],
            "stats": self.stats().model_dump(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryManifest":
        """Deserialize manifest."""
        manifest = cls(
            session_id=data["session_id"],
        )
        manifest.id = UUID(data["id"])
        manifest.created_at = datetime.fromisoformat(data["created_at"])

        # Reconstruct operations
        type_map = {
            "externalize": ExternalizeOperation,
            "deduplicate": DeduplicateOperation,
            "collapse": CollapseOperation,
            "schema": SchemaOperation,
            "entity_compress": EntityCompressOperation,
            "summarize": SummarizeOperation,
        }

        for op_data in data.get("operations", []):
            op_type = op_data.get("type")
            if op_type in type_map:
                op = type_map[op_type].model_validate(op_data)
                manifest.log_operation(op)

        return manifest
```

---

## Task Breakdown

### Week 7-8: Pipeline Framework

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| CP-001 | Create `types.py` with all shared types | 3 | None | All models validate |
| CP-002 | Define `CompressionStrategy` protocol | 2 | CP-001 | Protocol complete |
| CP-003 | Implement `PreservationRule` logic | 3 | CP-001 | Rules filter nodes |
| CP-004 | Implement `CompressionPipeline` core | 6 | CP-002, CP-003 | Strategy ordering works |
| CP-005 | Implement `plan()` method | 4 | CP-004 | Dry-run estimates |
| CP-006 | Implement `compress()` method | 6 | CP-004 | Full compression works |
| CP-007 | Implement `compress_to_budget()` | 3 | CP-006 | Budget-based compression |
| CP-008 | Write unit tests | 8 | CP-001 to CP-007 | 90%+ coverage |

### Week 9: Lossless Strategies

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| LS-001 | Define `ExternalStorage` protocol | 2 | None | Protocol complete |
| LS-002 | Implement `InMemoryExternalStorage` | 2 | LS-001 | Store/retrieve works |
| LS-003 | Implement `ExternalizePayloads` | 8 | LS-001, LS-002 | Large payloads externalized |
| LS-004 | Implement `DeduplicateSemantically` | 6 | context-core SemanticIndex | Duplicates removed |
| LS-005 | Implement `CollapseToolChains` | 6 | None | Tool chains collapsed |
| LS-006 | Write unit tests | 8 | LS-001 to LS-005 | 90%+ coverage |

### Week 10: Compaction Strategies

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| CS-001 | Implement `SchemaCompression` | 8 | None | Schemas extracted |
| CS-002 | Implement `EntityCentricCompression` | 8 | context-core EntityTracker | Entity context preserved |
| CS-003 | Implement `TaskRelevanceCompression` | 6 | None | Off-task content removed |
| CS-004 | Write unit tests | 8 | CS-001 to CS-003 | 90%+ coverage |

### Week 11: Summarization & Recovery

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| SS-001 | Define `LLMSummarizer` protocol | 2 | None | Protocol complete |
| SS-002 | Implement `HierarchicalSummarization` | 8 | SS-001 | Hierarchical summaries |
| SS-003 | Implement `TaskAwareSummarization` | 6 | SS-001 | Task-focused summaries |
| SS-004 | Implement operation types | 4 | None | All operations defined |
| SS-005 | Implement `RecoveryManifest` | 6 | SS-004 | Recovery tracking works |
| SS-006 | Write unit tests | 8 | SS-001 to SS-005 | 90%+ coverage |

### Week 12: Integration

| Task ID | Task | Est. Hours | Dependencies | Acceptance Criteria |
|---------|------|------------|--------------|---------------------|
| INT-001 | Create public API | 3 | All above | Clean exports |
| INT-002 | Write integration tests | 8 | INT-001 | Full pipeline works |
| INT-003 | Performance benchmarks | 4 | INT-001 | Baseline established |
| INT-004 | Documentation | 6 | INT-001 | API docs complete |

---

## Test Specifications

### Example Test Cases

```python
# tests/test_compression_pipeline.py
import pytest
from context_core.graph import ContextGraph
from context_compression import (
    CompressionPipeline, CompressionTier,
    ExternalizePayloads, DeduplicateSemantically,
)

class TestCompressionPipeline:
    def test_preservation_rules_protect_recent(self):
        graph = ContextGraph()
        for i in range(20):
            graph.add_message(role="user", content=f"Message {i}")

        pipeline = CompressionPipeline(preserve_recent_n=10)
        preserved = pipeline.get_preserved_nodes(graph)

        # Last 10 should be preserved
        recent_10 = graph.get_recent(10)
        for node in recent_10:
            assert node.id in preserved

    def test_strategies_ordered_by_tier(self):
        pipeline = CompressionPipeline()
        pipeline.register_strategy(MockSummarizationStrategy())
        pipeline.register_strategy(MockLosslessStrategy())
        pipeline.register_strategy(MockCompactionStrategy())

        # Should be ordered: lossless, compaction, summarization
        assert pipeline._strategies[0].tier == CompressionTier.LOSSLESS
        assert pipeline._strategies[1].tier == CompressionTier.COMPACTION
        assert pipeline._strategies[2].tier == CompressionTier.SUMMARIZATION

    def test_compress_respects_target_tokens(self):
        graph = ContextGraph()
        for i in range(100):
            node = graph.add_message(role="user", content="x" * 1000)
            node.token_count = 250

        pipeline = CompressionPipeline(strategies=[MockLosslessStrategy()])
        results = pipeline.compress(graph, target_tokens=5000)

        total_saved = sum(r.tokens_saved for r in results)
        assert total_saved >= 5000
        assert total_saved < 10000  # Should stop near target


class TestExternalizePayloads:
    def test_externalizes_large_tool_results(self):
        from context_compression.strategies.lossless.externalize import (
            ExternalizePayloads, InMemoryExternalStorage
        )

        graph = ContextGraph()
        call = graph.add_tool_call("read_file", {"path": "/test"})
        result = graph.add_tool_result(call.id, "x" * 5000)
        result.token_count = 1250

        storage = InMemoryExternalStorage()
        strategy = ExternalizePayloads(storage, min_tokens=500)
        manifest = RecoveryManifest(session_id="test")

        result = strategy.compress(graph, manifest)

        assert result.tokens_saved > 1000
        assert result.nodes_compressed == 1
        assert result.is_recoverable

    def test_preserves_preview(self):
        # ... test that preview is kept
        pass


class TestRecoveryManifest:
    def test_can_recover_lossless_operations(self):
        manifest = RecoveryManifest(session_id="test")
        manifest.log_operation(ExternalizeOperation(
            node_id=uuid4(),
            external_uri="memory://test",
            original_tokens=1000,
            original_content_hash="abc123",
        ))

        assert manifest.can_recover_node(manifest._operations[0].node_id)

    def test_cannot_recover_summarized(self):
        manifest = RecoveryManifest(session_id="test")
        node_id = uuid4()
        manifest.log_operation(SummarizeOperation(
            node_id=node_id,
            original_node_ids=[uuid4(), uuid4()],
            summary_node_id=uuid4(),
            original_tokens=1000,
            summary_tokens=100,
            method="hierarchical",
        ))

        assert not manifest.can_recover_node(node_id)
```

---

## Definition of Done for Phase 2

1. **All 8 strategies implemented** and tested
2. **Recovery manifest** tracks all operations
3. **Unit test coverage** >= 90%
4. **Integration tests** with context-core pass
5. **Compression ratios** meet targets:
   - Lossless: 2-3x
   - Compaction: 3-5x
   - Summarization: 5-10x
6. **Performance benchmarks** established:
   - 10,000 nodes compressed in < 5s
   - Recovery lookup in < 10ms
7. **Documentation** complete for all strategies

---

*This specification provides complete, implementable code for the compression pipeline. Each strategy can be implemented independently and tested in isolation before integration.*
