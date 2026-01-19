"""Ensemble retrieval combining multiple strategies.

Uses Reciprocal Rank Fusion (RRF) to combine rankings from
multiple retrieval strategies with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from context_memory.retrieval.base import RetrievalQuery, RetrievalStrategy
from context_memory.types import RetrievalResult

if TYPE_CHECKING:
    from context_memory.store import MemoryStore


@dataclass
class StrategyConfig:
    """Configuration for a strategy in the ensemble.

    Attributes:
        strategy: The retrieval strategy instance
        weight: Relative weight for RRF scoring (0-1)
        enabled: Whether this strategy is active
    """

    strategy: RetrievalStrategy
    weight: float = 1.0
    enabled: bool = True


@dataclass
class EnsembleResult:
    """Intermediate result for ensemble scoring."""

    result: RetrievalResult
    strategy_ranks: dict[str, int] = field(default_factory=dict)
    combined_score: float = 0.0


class EnsembleRetriever:
    """Combine multiple retrieval strategies using Reciprocal Rank Fusion.

    RRF combines rankings from multiple strategies into a unified ranking
    that benefits from the strengths of each approach. The formula is:
        RRF(d) = sum(weight_i / (k + rank_i(d)))

    where k is a constant (default 60) that dampens the impact of high
    rankings.

    Example:
        >>> ensemble = EnsembleRetriever(
        ...     strategies=[
        ...         (semantic_strategy, 1.0),
        ...         (entity_strategy, 0.8),
        ...         (temporal_strategy, 0.5),
        ...     ],
        ...     k=60,
        ... )
        >>> results = await ensemble.retrieve(query, store)
    """

    def __init__(
        self,
        strategies: list[tuple[RetrievalStrategy, float]],
        k: int = 60,
    ) -> None:
        """Initialize EnsembleRetriever.

        Args:
            strategies: List of (strategy, weight) tuples.
                Weights are relative and will be normalized.
            k: RRF constant. Higher values reduce the impact of
                top rankings. Default 60 is standard.
        """
        self._strategies = [StrategyConfig(strategy=s, weight=w) for s, w in strategies]
        self._k = k

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "ensemble"

    def add_strategy(
        self,
        strategy: RetrievalStrategy,
        weight: float = 1.0,
    ) -> None:
        """Add a new strategy to the ensemble.

        Args:
            strategy: Retrieval strategy to add
            weight: Relative weight for this strategy
        """
        self._strategies.append(StrategyConfig(strategy=strategy, weight=weight))

    def set_strategy_enabled(self, strategy_name: str, enabled: bool) -> None:
        """Enable or disable a strategy by name.

        Args:
            strategy_name: Name of the strategy to modify
            enabled: Whether the strategy should be active
        """
        for config in self._strategies:
            if config.strategy.name == strategy_name:
                config.enabled = enabled
                break

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Any | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using all enabled strategies and combine rankings.

        Args:
            query: Retrieval query parameters
            store: Memory store to query
            current_context: Passed to underlying strategies

        Returns:
            Combined results ranked by RRF score
        """
        # Collect results from all enabled strategies
        node_results: dict[UUID, EnsembleResult] = {}

        for config in self._strategies:
            if not config.enabled:
                continue

            strategy = config.strategy
            weight = config.weight

            try:
                results = await strategy.retrieve(query, store, current_context)
            except Exception:
                # Skip failed strategies but continue with others
                continue

            for rank, result in enumerate(results):
                node_id = result.node.id

                if node_id not in node_results:
                    node_results[node_id] = EnsembleResult(result=result)

                ensemble_result = node_results[node_id]
                ensemble_result.strategy_ranks[strategy.name] = rank

                # RRF score contribution
                rrf_score = weight / (self._k + rank + 1)
                ensemble_result.combined_score += rrf_score

        if not node_results:
            return []

        # Sort by combined score
        sorted_results = sorted(
            node_results.values(),
            key=lambda x: x.combined_score,
            reverse=True,
        )

        # Build final results with updated scores
        final_results: list[RetrievalResult] = []
        for ensemble_result in sorted_results[: query.max_results]:
            result = ensemble_result.result
            # Normalize score to [0, 1] range
            # Max possible score is sum of weights / (k + 1)
            max_score = sum(
                c.weight / (self._k + 1) for c in self._strategies if c.enabled
            )
            normalized_score = (
                ensemble_result.combined_score / max_score if max_score > 0 else 0.0
            )

            final_results.append(
                RetrievalResult(
                    node=result.node,
                    score=min(normalized_score, 1.0),
                    source_tier=result.source_tier,
                    retrieval_method=self.name,
                    latency_ms=result.latency_ms,
                )
            )

        return final_results


class MemoryRetriever:
    """High-level retriever with sensible defaults.

    Convenience class that sets up an EnsembleRetriever with
    commonly used strategies. Provides a simpler interface for
    typical retrieval scenarios.

    Example:
        >>> retriever = MemoryRetriever(store=my_store)
        >>> retriever.add_semantic(semantic_index)
        >>> retriever.add_entity(entity_tracker)
        >>> results = await retriever.search("authentication flow")
    """

    def __init__(
        self,
        store: MemoryStore,
        default_session_id: str | None = None,
    ) -> None:
        """Initialize MemoryRetriever.

        Args:
            store: Memory store to search
            default_session_id: Default session ID for queries
        """
        self._store = store
        self._default_session_id = default_session_id
        self._ensemble = EnsembleRetriever(strategies=[])

    def add_semantic(
        self,
        semantic_index: Any,  # SemanticIndex
        weight: float = 1.0,
        min_similarity: float = 0.6,
    ) -> None:
        """Add semantic retrieval strategy.

        Args:
            semantic_index: Populated SemanticIndex
            weight: Weight for ensemble scoring
            min_similarity: Minimum similarity threshold
        """
        from context_memory.retrieval.semantic import SemanticRetrieval

        strategy = SemanticRetrieval(
            semantic_index=semantic_index,
            min_similarity=min_similarity,
        )
        self._ensemble.add_strategy(strategy, weight)

    def add_entity(
        self,
        entity_tracker: Any,  # EntityTracker
        weight: float = 0.8,
    ) -> None:
        """Add entity-based retrieval strategy.

        Args:
            entity_tracker: Populated EntityTracker
            weight: Weight for ensemble scoring
        """
        from context_memory.retrieval.entity import EntityRetrieval

        strategy = EntityRetrieval(entity_tracker=entity_tracker)
        self._ensemble.add_strategy(strategy, weight)

    def add_temporal(
        self,
        weight: float = 0.5,
        default_window_hours: float = 24.0,
    ) -> None:
        """Add temporal retrieval strategy.

        Args:
            weight: Weight for ensemble scoring
            default_window_hours: Default time window
        """
        from context_memory.retrieval.temporal import TemporalRetrieval

        strategy = TemporalRetrieval(default_window_hours=default_window_hours)
        self._ensemble.add_strategy(strategy, weight)

    async def search(
        self,
        query_text: str,
        session_id: str | None = None,
        max_results: int = 10,
        **filters: Any,
    ) -> list[RetrievalResult]:
        """Search for relevant context.

        Args:
            query_text: Text to search for
            session_id: Session to search (uses default if not provided)
            max_results: Maximum results to return
            **filters: Additional filters (node_types, min_importance, etc.)

        Returns:
            List of relevant results

        Raises:
            ValueError: If no session_id provided and no default set
        """
        sid = session_id or self._default_session_id
        if not sid:
            raise ValueError("session_id required (no default set)")

        query = RetrievalQuery(
            session_id=sid,
            query_text=query_text,
            max_results=max_results,
            **filters,
        )

        return await self._ensemble.retrieve(query, self._store)
