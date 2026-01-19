"""Temporal retrieval strategy based on time windows.

Retrieves context within specified time ranges with recency scoring.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from context_memory.retrieval.base import RetrievalQuery
from context_memory.types import RetrievalResult, StorageTier

if TYPE_CHECKING:
    from context_memory.store import MemoryStore


class TemporalRetrieval:
    """Retrieve context based on temporal proximity.

    Finds recent context within a time window, scoring by recency.
    Useful for retrieving recent conversation history or finding
    context from a specific time period.

    Example:
        >>> strategy = TemporalRetrieval(default_window_hours=24)
        >>> results = await strategy.retrieve(query, store)
    """

    def __init__(
        self,
        default_window_hours: float = 24.0,
        recency_decay: float = 0.5,
    ) -> None:
        """Initialize TemporalRetrieval.

        Args:
            default_window_hours: Default time window if not specified in query
            recency_decay: Decay factor for recency scoring (0-1).
                Higher values give more weight to very recent items.
        """
        self._default_window_hours = default_window_hours
        self._recency_decay = recency_decay

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "temporal"

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Any | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve based on temporal proximity.

        Uses query.since and query.until if provided, otherwise uses
        default time window from now.

        Args:
            query: May include since/until for time bounds
            store: Memory store containing the nodes
            current_context: Ignored for temporal retrieval

        Returns:
            List of recent nodes, scored by recency
        """
        start = time.perf_counter()
        now = datetime.now(UTC)

        # Determine time window
        since = query.since
        until = query.until or now

        if since is None:
            since = now - timedelta(hours=self._default_window_hours)

        # Get all keys in session
        keys = await store.list_keys(session_id=query.session_id)

        results: list[RetrievalResult] = []
        window_duration = (until - since).total_seconds()

        for key in keys:
            metadata = await store.get_metadata(key)
            if metadata is None:
                continue

            # Check time bounds
            if metadata.created_at < since or metadata.created_at > until:
                continue

            # Apply query filters
            if query.node_types and metadata.node_type not in query.node_types:
                continue

            if (
                query.min_importance is not None
                and metadata.importance < query.min_importance
            ):
                continue

            # Retrieve full node
            node = await store.retrieve(key)
            if node is None:
                continue

            # Calculate recency score
            age_seconds = (until - metadata.created_at).total_seconds()
            if window_duration > 0:
                # Exponential decay based on age
                recency_factor = 1.0 - (age_seconds / window_duration)
                score = recency_factor**self._recency_decay
            else:
                score = 1.0

            # Blend with importance
            score = 0.7 * score + 0.3 * metadata.importance

            latency_ms = (time.perf_counter() - start) * 1000

            results.append(
                RetrievalResult(
                    node=node,
                    score=score,
                    source_tier=StorageTier.WARM,
                    retrieval_method=self.name,
                    latency_ms=latency_ms,
                )
            )

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[: query.max_results]
