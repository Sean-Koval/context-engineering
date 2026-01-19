"""Semantic retrieval strategy using embedding similarity.

Uses SemanticIndex from context-core to find contextually similar
past interactions based on embedding vectors.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from context_memory.retrieval.base import RetrievalQuery
from context_memory.types import RetrievalResult, StorageKey, StorageTier

if TYPE_CHECKING:
    from context_core.semantic import SemanticIndex
    from context_memory.store import MemoryStore


class SemanticRetrieval:
    """Retrieve context using semantic similarity.

    Uses embeddings to find past context that is semantically similar
    to the query text. Requires a SemanticIndex that has been populated
    with node embeddings.

    Example:
        >>> from context_core.semantic import SemanticIndex
        >>> index = SemanticIndex()
        >>> strategy = SemanticRetrieval(semantic_index=index, min_similarity=0.7)
        >>> results = await strategy.retrieve(query, store)
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        min_similarity: float = 0.6,
    ) -> None:
        """Initialize SemanticRetrieval.

        Args:
            semantic_index: Populated SemanticIndex for similarity search
            min_similarity: Minimum similarity score threshold [0, 1]
        """
        self._semantic_index = semantic_index
        self._min_similarity = min_similarity

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "semantic"

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Any | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve using semantic similarity.

        Args:
            query: Must include query_text for semantic search
            store: Memory store containing the nodes
            current_context: Ignored for semantic retrieval

        Returns:
            List of semantically similar nodes, sorted by similarity
        """
        if not query.query_text:
            return []

        start = time.perf_counter()

        # Search semantic index for similar content
        search_results = self._semantic_index.search(
            query=query.query_text,
            k=query.max_results * 2,  # Get extra for filtering
            min_score=self._min_similarity,
        )

        results: list[RetrievalResult] = []

        for sr in search_results:
            # Build storage key to retrieve full node
            key = StorageKey(
                session_id=query.session_id,
                node_id=sr.id,
                version=1,
            )

            node = await store.retrieve(key)
            if node is None:
                continue

            # Apply query filters
            if query.node_types and node.type.value not in query.node_types:
                continue

            if (
                query.min_importance is not None
                and node.metadata.importance < query.min_importance
            ):
                continue

            latency_ms = (time.perf_counter() - start) * 1000

            results.append(
                RetrievalResult(
                    node=node,
                    score=sr.score,
                    source_tier=StorageTier.WARM,
                    retrieval_method=self.name,
                    latency_ms=latency_ms,
                )
            )

            if len(results) >= query.max_results:
                break

        return results
