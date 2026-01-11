"""Semantic deduplication strategy.

This strategy identifies and removes semantically duplicate nodes from the
context graph, keeping only a canonical version of each group of duplicates.
All edges are redirected to the canonical node to maintain graph integrity.
"""

from __future__ import annotations

import contextlib
import json
import time
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.recovery import RecoveryManifest
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel

if TYPE_CHECKING:
    from context_core.graph import ContextGraph, ContextNode
    from context_core.semantic import SemanticIndex


class DeduplicateSemantically(BaseCompressionStrategy):
    """Remove semantically duplicate nodes from the context graph.

    This strategy uses semantic similarity (via embeddings) to identify
    near-duplicate content across nodes. When duplicates are found, it:

    1. Groups all duplicates using transitive closure (if A~B and B~C, all
       three are grouped together)
    2. Selects the best "canonical" node from each group based on:
       - Recency (prefer more recent nodes)
       - Importance score (prefer higher importance)
       - Token count (prefer more detailed content)
    3. Redirects all edges from removed nodes to the canonical
    4. Logs operations for recovery

    Targets:
        - Repeated tool outputs
        - Similar message content
        - Redundant information

    Preserves:
        - Most recent/important version as canonical
        - All edge relationships (redirected to canonical)
        - Original content in recovery manifest

    Configuration:
        similarity_threshold: Minimum similarity to consider duplicate (0.92)
        min_tokens_to_dedupe: Don't dedupe nodes below this threshold (50)
        prefer_recent: Weight recent nodes higher for canonical selection

    The strategy is fully reversible - original content is stored in the
    recovery manifest.
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        similarity_threshold: float = 0.92,
        min_tokens_to_dedupe: int = 50,
        prefer_recent: bool = True,
    ) -> None:
        """Initialize the deduplication strategy.

        Args:
            semantic_index: The semantic index containing node embeddings.
            similarity_threshold: Minimum similarity score (0-1) to consider
                two nodes as duplicates. Higher values are more conservative.
            min_tokens_to_dedupe: Minimum token count for a node to be
                considered for deduplication. Tiny nodes are skipped.
            prefer_recent: If True, prefer more recent nodes as canonical.
        """
        self._semantic_index = semantic_index
        self._similarity_threshold = similarity_threshold
        self._min_tokens = min_tokens_to_dedupe
        self._prefer_recent = prefer_recent

    @property
    def _name(self) -> str:
        return "deduplicate_semantically"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.LOSSLESS

    @property
    def _priority(self) -> int:
        return 20  # Run after externalize (10), before collapse (20+)

    def _find_duplicate_groups(
        self,
        duplicates: list[tuple[UUID, UUID, float]],
        graph: ContextGraph,
    ) -> dict[UUID, list[tuple[UUID, float]]]:
        """Build duplicate groups using transitive closure.

        When A~B and B~C, all three should be in the same group. This method
        uses a union-find approach to compute the transitive closure of the
        similarity relationship.

        Args:
            duplicates: List of (id1, id2, similarity) tuples from find_duplicates.
            graph: The context graph for node lookups.

        Returns:
            Dictionary mapping canonical_id to list of (duplicate_id, similarity)
            tuples. The canonical is selected using _select_canonical.
        """
        if not duplicates:
            return {}

        # Build adjacency list for transitive closure
        adjacency: dict[UUID, set[UUID]] = {}
        similarity_map: dict[tuple[UUID, UUID], float] = {}

        for id1, id2, score in duplicates:
            # Check both nodes exist and are eligible
            node1 = graph.get_node(id1)
            node2 = graph.get_node(id2)

            if not node1 or not node2:
                continue

            # Skip if either node is too small
            if (node1.token_count or 0) < self._min_tokens:
                continue
            if (node2.token_count or 0) < self._min_tokens:
                continue

            # Skip pinned nodes
            if node1.metadata.pinned or node2.metadata.pinned:
                continue

            # Skip already compressed nodes
            if node1.compression_level != CompressionLevel.FULL:
                continue
            if node2.compression_level != CompressionLevel.FULL:
                continue

            # Add to adjacency
            adjacency.setdefault(id1, set()).add(id2)
            adjacency.setdefault(id2, set()).add(id1)

            # Store similarity (both directions for easy lookup)
            similarity_map[(id1, id2)] = score
            similarity_map[(id2, id1)] = score

        if not adjacency:
            return {}

        # Find connected components using BFS (transitive closure)
        visited: set[UUID] = set()
        components: list[set[UUID]] = []

        for start_id in adjacency:
            if start_id in visited:
                continue

            # BFS to find all connected nodes
            component: set[UUID] = set()
            queue = [start_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 2:
                components.append(component)

        # Select canonical for each component and build result
        result: dict[UUID, list[tuple[UUID, float]]] = {}

        for component in components:
            canonical_id = self._select_canonical(component, graph)
            duplicates_list: list[tuple[UUID, float]] = []

            for node_id in component:
                if node_id == canonical_id:
                    continue

                # Get similarity score with canonical
                score = similarity_map.get((canonical_id, node_id), 0.0)
                if score == 0.0:
                    # If no direct edge, estimate from any connected node
                    for other_id in component:
                        if other_id != node_id:
                            score = similarity_map.get((other_id, node_id), 0.0)
                            if score > 0:
                                break

                duplicates_list.append((node_id, score))

            if duplicates_list:
                result[canonical_id] = duplicates_list

        return result

    def _select_canonical(
        self,
        group: set[UUID],
        graph: ContextGraph,
    ) -> UUID:
        """Select the best canonical node from a group.

        Selection criteria (in order of importance):
        1. Higher importance score (40% weight)
        2. More recent (30% weight) - if prefer_recent is True
        3. More tokens (30% weight) - prefer detailed content

        Args:
            group: Set of node IDs in the duplicate group.
            graph: The context graph for node lookups.

        Returns:
            UUID of the selected canonical node.
        """
        scores: dict[UUID, float] = {}

        # Collect metrics for all nodes
        nodes: list[tuple[UUID, ContextNode]] = []
        for node_id in group:
            node = graph.get_node(node_id)
            if node:
                nodes.append((node_id, node))

        if not nodes:
            # Fallback: return any ID from the group
            return next(iter(group))

        if len(nodes) == 1:
            return nodes[0][0]

        # Normalize metrics for scoring
        max_importance = max(n.compute_importance() for _, n in nodes)
        max_sequence = max((n.sequence_number or 0) for _, n in nodes)
        min_sequence = min((n.sequence_number or 0) for _, n in nodes)
        max_tokens = max((n.token_count or 0) for _, n in nodes)

        seq_range = max_sequence - min_sequence if max_sequence > min_sequence else 1

        for node_id, node in nodes:
            # Importance score (0-1)
            importance_score = (
                node.compute_importance() / max_importance
                if max_importance > 0
                else 0.5
            )

            # Recency score (0-1), higher for more recent
            sequence = node.sequence_number or 0
            recency_score = (sequence - min_sequence) / seq_range

            # Token score (0-1), prefer more detailed content
            token_score = (
                (node.token_count or 0) / max_tokens if max_tokens > 0 else 0.5
            )

            # Combine scores
            if self._prefer_recent:
                combined = (
                    0.4 * importance_score + 0.3 * recency_score + 0.3 * token_score
                )
            else:
                combined = 0.5 * importance_score + 0.5 * token_score

            scores[node_id] = combined

        # Return highest scoring node
        return max(scores.keys(), key=lambda x: scores[x])

    def _serialize_node_content(self, node: ContextNode) -> str:
        """Serialize node content for recovery.

        Args:
            node: The node to serialize.

        Returns:
            JSON string of the node's content.
        """
        content_dict = node.content.model_dump(mode="json")
        return json.dumps(content_dict)

    def _redirect_edges(
        self,
        graph: ContextGraph,
        from_id: UUID,
        to_id: UUID,
    ) -> None:
        """Redirect all edges from one node to another.

        This preserves graph connectivity when removing duplicate nodes.
        Incoming edges to from_id are redirected to to_id.
        Outgoing edges from from_id are handled by the node removal.

        Args:
            graph: The context graph.
            from_id: ID of the node being removed.
            to_id: ID of the canonical node to redirect to.
        """
        from context_core.graph.edges import Edge

        # Get incoming edges to the node being removed
        incoming_edges = graph.get_edges(target_id=from_id)

        for edge in incoming_edges:
            # Skip self-loops and edges from the canonical
            if edge.source_id == from_id or edge.source_id == to_id:
                continue

            # Create new edge to canonical (if not already exists)
            existing = graph.get_edges(source_id=edge.source_id, target_id=to_id)
            has_same_type = any(e.type == edge.type for e in existing)

            if not has_same_type:
                new_edge = Edge(
                    source_id=edge.source_id,
                    target_id=to_id,
                    type=edge.type,
                    metadata=edge.metadata,
                )
                # Edge might fail if source doesn't exist anymore
                with contextlib.suppress(ValueError):
                    graph.add_edge(new_edge)

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved by deduplication.

        Args:
            graph: The context graph to analyze.
            target_node_ids: Optional list of specific node IDs to target.

        Returns:
            Estimated number of tokens that would be saved.
        """
        duplicates = self._semantic_index.find_duplicates(
            threshold=self._similarity_threshold
        )

        if not duplicates:
            return 0

        # Build duplicate groups
        groups = self._find_duplicate_groups(duplicates, graph)

        if not groups:
            return 0

        target_set = set(target_node_ids) if target_node_ids else None
        total_savings = 0

        for _canonical_id, dupe_list in groups.items():
            for dupe_id, _ in dupe_list:
                # Skip if not in target set
                if target_set and dupe_id not in target_set:
                    continue

                node = graph.get_node(dupe_id)
                if node:
                    total_savings += node.token_count or 0

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute semantic deduplication on the graph.

        Args:
            graph: The context graph to compress (modified in place).
            manifest: Recovery manifest to log operations.
            target_node_ids: Optional list of specific node IDs to target.
            target_tokens: Stop when this many tokens have been saved.

        Returns:
            CompressionResult with metrics about the operation.
        """
        start_time = time.perf_counter()

        # Find duplicate pairs
        duplicates = self._semantic_index.find_duplicates(
            threshold=self._similarity_threshold
        )

        if not duplicates:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=len(graph),
                is_recoverable=True,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Build duplicate groups with transitive closure
        groups = self._find_duplicate_groups(duplicates, graph)

        if not groups:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=len(graph),
                is_recoverable=True,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )

        target_set = set(target_node_ids) if target_node_ids else None

        original_tokens = 0
        tokens_saved = 0
        nodes_removed = 0
        removed_ids: set[UUID] = set()

        # Process each group
        for canonical_id, dupe_list in groups.items():
            # Sort by similarity descending for deterministic behavior
            sorted_dupes = sorted(dupe_list, key=lambda x: x[1], reverse=True)

            for dupe_id, similarity in sorted_dupes:
                # Check target tokens limit
                if target_tokens and tokens_saved >= target_tokens:
                    break

                # Skip if already removed
                if dupe_id in removed_ids:
                    continue

                # Skip if not in target set
                if target_set and dupe_id not in target_set:
                    continue

                dupe_node = graph.get_node(dupe_id)
                if not dupe_node:
                    continue

                dupe_tokens = dupe_node.token_count or 0
                original_tokens += dupe_tokens

                # Serialize content for recovery
                original_content = self._serialize_node_content(dupe_node)

                # Log operation to manifest
                # Clamp similarity to [0, 1] to handle floating point precision issues
                clamped_similarity = min(1.0, max(0.0, similarity))
                manifest.log_deduplicate(
                    node_id=dupe_id,
                    removed_node_ids=[dupe_id],
                    kept_node_id=canonical_id,
                    original_tokens=dupe_tokens,
                    similarity_score=clamped_similarity,
                    original_contents={str(dupe_id): original_content},
                )

                # Redirect edges before removal
                self._redirect_edges(graph, dupe_id, canonical_id)

                # Remove from graph and semantic index
                graph.remove_node(dupe_id)
                self._semantic_index.remove_node(dupe_id)

                removed_ids.add(dupe_id)
                nodes_removed += 1
                tokens_saved += dupe_tokens

            # Check if we've hit the target
            if target_tokens and tokens_saved >= target_tokens:
                break

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=0,  # Duplicates are fully removed
            tokens_saved=tokens_saved,
            nodes_processed=len(graph) + nodes_removed,
            nodes_compressed=0,
            nodes_removed=nodes_removed,
            nodes_created=0,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=True,
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if deduplication can be applied.

        Returns True if there are at least 2 nodes and the semantic index
        has enough indexed content to find duplicates.

        Args:
            graph: The context graph to check.

        Returns:
            True if deduplication can potentially be applied.
        """
        if len(graph) < 2:
            return False

        # Check if semantic index has enough content
        if len(self._semantic_index) < 2:
            return False

        # Quick check: are there any duplicates at all?
        duplicates = self._semantic_index.find_duplicates(
            threshold=self._similarity_threshold
        )

        return len(duplicates) > 0
