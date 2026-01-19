"""Entity-based retrieval strategy.

Uses EntityTracker from context-core to find past context
mentioning the same entities as the query.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from context_memory.retrieval.base import RetrievalQuery
from context_memory.types import RetrievalResult, StorageKey, StorageTier

if TYPE_CHECKING:
    from context_core.entities import EntityTracker
    from context_memory.store import MemoryStore


class EntityRetrieval:
    """Retrieve context based on entity mentions.

    Finds past context that mentions the same entities as specified
    in the query or extracted from the query text. Useful for finding
    related discussions about specific people, systems, or concepts.

    Example:
        >>> from context_core.entities import EntityTracker
        >>> tracker = EntityTracker()
        >>> strategy = EntityRetrieval(entity_tracker=tracker)
        >>> results = await strategy.retrieve(query, store)
    """

    def __init__(self, entity_tracker: EntityTracker) -> None:
        """Initialize EntityRetrieval.

        Args:
            entity_tracker: Populated EntityTracker for entity lookups
        """
        self._entity_tracker = entity_tracker

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "entity"

    async def retrieve(
        self,
        query: RetrievalQuery,
        store: MemoryStore,
        current_context: Any | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve based on entity mentions.

        If entity_ids not provided but query_text is, extracts entities
        from the query text first.

        Args:
            query: Should include entity_ids or query_text
            store: Memory store containing the nodes
            current_context: Ignored for entity retrieval

        Returns:
            List of nodes mentioning the same entities, scored by
            entity importance and mention frequency
        """
        entity_ids = list(query.entity_ids) if query.entity_ids else []

        # Extract entities from query text if no explicit IDs
        if not entity_ids and query.query_text:
            from uuid import uuid4

            temp_node_id = uuid4()
            entities = self._entity_tracker.extract_from_text(
                query.query_text,
                temp_node_id,
            )
            entity_ids = [str(e.id) for e in entities]

        if not entity_ids:
            return []

        start = time.perf_counter()
        results: list[RetrievalResult] = []
        seen_node_ids: set[UUID] = set()

        for entity_id in entity_ids:
            try:
                entity = self._entity_tracker.get_entity(UUID(entity_id))
            except ValueError:
                continue

            if entity is None:
                continue

            for node_id in entity.node_ids:
                # Skip duplicates
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)

                key = StorageKey(
                    session_id=query.session_id,
                    node_id=node_id,
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

                # Score based on entity importance and mention count
                entity_importance = entity.importance
                mention_count = entity.mention_count
                score = 0.5 * entity_importance + 0.5 * min(mention_count / 10, 1.0)

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
