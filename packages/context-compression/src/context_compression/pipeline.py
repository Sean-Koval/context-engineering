"""Compression pipeline orchestrator.

The CompressionPipeline coordinates multiple compression strategies,
applying them in tier order (LOSSLESS -> COMPACTION -> SUMMARIZATION)
while respecting preservation rules and target token budgets.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.base import CompressionStrategy
from context_compression.types import (
    CompressionPlan,
    CompressionResult,
    CompressionTier,
    PipelineConfig,
    PreservationRule,
)

if TYPE_CHECKING:
    from context_core.graph import ContextGraph, ContextNode


# Tier ordering for strategy execution
_TIER_ORDER = {
    CompressionTier.LOSSLESS: 0,
    CompressionTier.COMPACTION: 1,
    CompressionTier.SUMMARIZATION: 2,
}


class CompressionPipeline:
    """Orchestrates multi-strategy compression.

    The pipeline:
    1. Identifies nodes that should be preserved
    2. Executes strategies in tier order (lossless first)
    3. Stops when target tokens saved or no more progress
    4. Logs all operations to a recovery manifest

    Default preservation rules protect:
    - Recent messages (last N)
    - Pinned content
    - System prompts
    - High-importance nodes

    Example:
        >>> pipeline = CompressionPipeline()
        >>> pipeline.register_strategy(ExternalizePayloads(storage))
        >>> pipeline.register_strategy(CollapseToolChains())
        >>> results = pipeline.compress(graph, target_tokens=5000)
        >>> print(f"Saved {sum(r.tokens_saved for r in results)} tokens")
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
        strategies: list[CompressionStrategy] | None = None,
        preservation_rules: list[PreservationRule] | None = None,
        config: PipelineConfig | None = None,
        on_compression: Callable[[CompressionResult], None] | None = None,
    ):
        """Initialize the compression pipeline.

        Args:
            strategies: Initial list of strategies to register
            preservation_rules: Rules for protecting nodes (uses defaults if None)
            config: Pipeline configuration
            on_compression: Callback invoked after each strategy completes
        """
        self._strategies: list[CompressionStrategy] = []
        self._preservation_rules = (
            preservation_rules
            if preservation_rules is not None
            else self.DEFAULT_PRESERVATION_RULES.copy()
        )
        self._config = config or PipelineConfig()
        self._on_compression = on_compression

        # Register initial strategies
        if strategies:
            for strategy in strategies:
                self.register_strategy(strategy)

    def _sort_strategies(self) -> None:
        """Sort strategies by tier then priority."""
        self._strategies.sort(key=lambda s: (_TIER_ORDER.get(s.tier, 99), s.priority))

    def register_strategy(self, strategy: CompressionStrategy) -> None:
        """Register a compression strategy.

        Strategies are automatically sorted by tier and priority.

        Args:
            strategy: The strategy to register
        """
        self._strategies.append(strategy)
        self._sort_strategies()

    def unregister_strategy(self, name: str) -> bool:
        """Remove a strategy by name.

        Args:
            name: Name of the strategy to remove

        Returns:
            True if strategy was removed, False if not found
        """
        for i, strategy in enumerate(self._strategies):
            if strategy.name == name:
                self._strategies.pop(i)
                return True
        return False

    def add_preservation_rule(self, rule: PreservationRule) -> None:
        """Add a preservation rule.

        Args:
            rule: The rule to add
        """
        self._preservation_rules.append(rule)
        # Sort by priority (higher first)
        self._preservation_rules.sort(key=lambda r: -r.priority)

    def remove_preservation_rule(self, name: str) -> bool:
        """Remove a preservation rule by name.

        Args:
            name: Name of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self._preservation_rules):
            if rule.name == name:
                self._preservation_rules.pop(i)
                return True
        return False

    def get_preserved_nodes(self, graph: ContextGraph) -> set[UUID]:
        """Get IDs of nodes that should be preserved from compression.

        Applies preservation rules and always preserves the most recent
        N nodes as configured.

        Args:
            graph: The context graph to analyze

        Returns:
            Set of node IDs that should be preserved
        """
        preserved: set[UUID] = set()
        now = datetime.now(UTC)

        # Always preserve most recent N nodes
        if self._config.preserve_recent_n > 0:
            recent = graph.get_recent(self._config.preserve_recent_n)
            preserved.update(n.id for n in recent)

        # Apply preservation rules
        for node in graph:
            for rule in self._preservation_rules:
                if self._matches_rule(node, rule, now):
                    preserved.add(node.id)
                    break  # Node is preserved, no need to check more rules

        return preserved

    def _matches_rule(
        self,
        node: ContextNode,
        rule: PreservationRule,
        now: datetime,
    ) -> bool:
        """Check if a node matches a preservation rule.

        A node matches if it satisfies ALL specified criteria in the rule.

        Args:
            node: The node to check
            rule: The preservation rule
            now: Current timestamp for age calculations

        Returns:
            True if the node should be preserved by this rule
        """
        # Pinned check
        if rule.pinned and node.metadata.pinned:
            return True

        # Node type check
        if rule.node_types is not None and node.type.value not in rule.node_types:
            return False

        # Importance check (uses raw importance, not composite score)
        if (
            rule.min_importance is not None
            and node.metadata.importance < rule.min_importance
        ):
            return False

        # Age check
        if rule.max_age_seconds is not None:
            age = (now - node.metadata.created_at).total_seconds()
            if age > rule.max_age_seconds:
                return False

        # Tag check
        if rule.required_tags is not None and not (
            rule.required_tags & node.metadata.tags
        ):
            return False

        # Entity check
        if rule.entity_ids is not None:
            node_entities = set(node.metadata.entities)
            if not any(e in node_entities for e in rule.entity_ids):
                return False

        # Custom predicate
        if rule.custom_predicate is not None and not rule.custom_predicate(node):
            return False

        # If we get here with any criteria specified, the node matches
        # If no criteria were specified, the rule doesn't match anything
        return any(
            (
                rule.pinned,
                rule.node_types is not None,
                rule.min_importance is not None,
                rule.max_age_seconds is not None,
                rule.required_tags is not None,
                rule.entity_ids is not None,
                rule.custom_predicate is not None,
            )
        )

    def plan(
        self,
        graph: ContextGraph,
        target_tokens: int | None = None,
        max_tier: CompressionTier = CompressionTier.SUMMARIZATION,
    ) -> CompressionPlan:
        """Create a compression plan without executing.

        Useful for previewing what compression would do.

        Args:
            graph: The context graph to analyze
            target_tokens: Target token savings
            max_tier: Maximum compression tier to include

        Returns:
            CompressionPlan with estimated results
        """
        preserved = self.get_preserved_nodes(graph)
        compressible = [n.id for n in graph if n.id not in preserved]
        max_tier_value = _TIER_ORDER.get(max_tier, 2)

        strategies_to_use = []
        estimated_total = 0

        for strategy in self._strategies:
            # Check tier limit
            if _TIER_ORDER.get(strategy.tier, 99) > max_tier_value:
                continue

            # Check if applicable
            if not strategy.can_apply(graph):
                continue

            # Estimate savings
            estimated = strategy.estimate_savings(graph, compressible)
            if estimated > 0:
                strategies_to_use.append(strategy.name)
                estimated_total += estimated

                # Stop if we've reached target
                if target_tokens and estimated_total >= target_tokens:
                    break

        # Collect preservation reasons
        preservation_reasons = []
        for rule in self._preservation_rules:
            matching = sum(
                1 for n in graph if self._matches_rule(n, rule, datetime.now(UTC))
            )
            if matching > 0:
                preservation_reasons.append(f"{rule.name}: {matching} nodes")

        if self._config.preserve_recent_n > 0:
            preservation_reasons.insert(
                0, f"recent_n: {min(len(graph), self._config.preserve_recent_n)} nodes"
            )

        return CompressionPlan(
            strategies=strategies_to_use,
            target_tokens=target_tokens,
            estimated_savings=estimated_total,
            nodes_affected=compressible,
            preserved_nodes=list(preserved),
            preservations=preservation_reasons,
        )

    def compress(
        self,
        graph: ContextGraph,
        target_tokens: int | None = None,
        max_tier: CompressionTier = CompressionTier.SUMMARIZATION,
        dry_run: bool = False,
        session_id: str = "",
    ) -> list[CompressionResult]:
        """Execute the compression pipeline.

        Applies strategies in tier order until target_tokens is reached
        or no more progress can be made.

        Args:
            graph: Context graph to compress (modified in place)
            target_tokens: Stop when this many tokens are saved
            max_tier: Maximum compression tier to use
            dry_run: If True, only estimate without modifying
            session_id: Session identifier for the recovery manifest

        Returns:
            List of CompressionResult for each strategy applied
        """
        if dry_run:
            plan = self.plan(graph, target_tokens, max_tier)
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
            session_id=session_id,
            enable_recovery=self._config.enable_recovery,
        )

        # Get preserved nodes
        preserved = self.get_preserved_nodes(graph)
        compressible = [n.id for n in graph if n.id not in preserved]

        max_tier_value = _TIER_ORDER.get(max_tier, 2)
        results: list[CompressionResult] = []
        total_saved = 0

        for strategy in self._strategies:
            # Check tier limit
            if _TIER_ORDER.get(strategy.tier, 99) > max_tier_value:
                continue

            # Check if strategy can apply
            if not strategy.can_apply(graph):
                continue

            # Check if we've reached target
            if target_tokens and total_saved >= target_tokens:
                break

            # Calculate remaining target
            remaining_target = target_tokens - total_saved if target_tokens else None

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

            except Exception as e:
                # Create error result
                result = CompressionResult(
                    success=False,
                    strategy_name=strategy.name,
                    tier=strategy.tier,
                    original_tokens=0,
                    compressed_tokens=0,
                    tokens_saved=0,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    is_recoverable=True,
                    error_message=str(e),
                )

                if self._config.stop_on_error:
                    results.append(result)
                    break

            results.append(result)
            total_saved += result.tokens_saved

            # Invoke callback
            if self._on_compression:
                self._on_compression(result)

            # Check minimum savings threshold
            if result.tokens_saved < self._config.min_savings_threshold:
                # Strategy made little progress, continue to next
                continue

            # Update compressible list (some nodes may have been removed)
            preserved = self.get_preserved_nodes(graph)
            compressible = [n.id for n in graph if n.id not in preserved]

        return results

    def compress_to_budget(
        self,
        graph: ContextGraph,
        budget_tokens: int,
        max_tier: CompressionTier = CompressionTier.SUMMARIZATION,
    ) -> list[CompressionResult]:
        """Compress until graph fits within a token budget.

        Args:
            graph: Context graph to compress
            budget_tokens: Maximum tokens allowed in the graph
            max_tier: Maximum compression tier to use

        Returns:
            List of CompressionResult for each strategy applied
        """
        # Calculate current token count
        current_tokens = sum(node.token_count or 0 for node in graph)

        if current_tokens <= budget_tokens:
            return []  # Already within budget

        target_savings = current_tokens - budget_tokens
        return self.compress(graph, target_tokens=target_savings, max_tier=max_tier)

    @property
    def strategies(self) -> list[CompressionStrategy]:
        """Get list of registered strategies (read-only copy)."""
        return list(self._strategies)

    @property
    def preservation_rules(self) -> list[PreservationRule]:
        """Get list of preservation rules (read-only copy)."""
        return list(self._preservation_rules)

    @property
    def config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config
