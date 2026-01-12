"""Incremental summarization strategy.

This strategy builds summaries incrementally as new messages arrive:
1. Maintains a running summary node
2. Periodically updates the summary with new content
3. Removes processed messages after incorporation
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.strategies.summarization.base import BaseSummarizationStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.edges import Edge
from context_core.graph.nodes import ContextNode
from context_core.graph.types import EdgeType, NodeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_compression.strategies.summarization.mock_summarizer import (
        LLMSummarizer,
    )
    from context_core.graph import ContextGraph


class IncrementalSummarization(BaseSummarizationStrategy):
    """Streaming summarization that builds incrementally.

    Unlike hierarchical summarization which processes chunks in batch,
    this strategy maintains a single running summary that gets updated
    as new messages are processed. This is ideal for continuous
    conversation contexts.

    Configuration:
        summarizer: LLM backend for generating summaries
        update_interval: Messages to accumulate before updating (default 5)
        max_summary_tokens: Maximum tokens in the running summary (default 500)

    The strategy:
    1. Finds or creates a running summary node
    2. Collects new messages since last update
    3. Generates updated summary incorporating new content
    4. Updates the running summary node
    5. Removes processed messages
    """

    # Tag used to identify running summary nodes
    RUNNING_SUMMARY_TAG = "incremental_running_summary"

    def __init__(
        self,
        summarizer: LLMSummarizer,
        update_interval: int = 5,
        max_summary_tokens: int = 500,
    ) -> None:
        """Initialize the incremental summarization strategy.

        Args:
            summarizer: LLM summarizer for generating summaries
            update_interval: Number of new messages before updating summary
            max_summary_tokens: Maximum tokens in the running summary
        """
        self._summarizer = summarizer
        self._update_interval = update_interval
        self._max_summary_tokens = max_summary_tokens

    @property
    def _name(self) -> str:
        return "incremental_summarization"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.SUMMARIZATION

    @property
    def _priority(self) -> int:
        return 32  # After hierarchical and task-aware

    def _find_running_summary(self, graph: ContextGraph) -> ContextNode | None:
        """Find existing incremental summary node.

        Looks for a SUMMARY node with the running summary tag.

        Args:
            graph: The context graph

        Returns:
            The running summary node or None if not found
        """
        for node in graph:
            if (
                node.type == NodeType.SUMMARY
                and self.RUNNING_SUMMARY_TAG in node.metadata.tags
            ):
                return node
        return None

    def _get_messages_since_summary(
        self,
        graph: ContextGraph,
        summary_node: ContextNode | None,
        target_node_ids: list[UUID] | None,
    ) -> list[ContextNode]:
        """Get MESSAGE nodes added after the summary node.

        Args:
            graph: The context graph
            summary_node: The existing summary node (or None)
            target_node_ids: Optional filter to specific nodes

        Returns:
            List of MESSAGE nodes newer than the summary
        """
        target_set = set(target_node_ids) if target_node_ids else None
        summary_seq = (summary_node.sequence_number if summary_node else -1) or -1

        # Get the IDs of nodes that were already summarized
        already_summarized: set[UUID] = set()
        if summary_node and summary_node.content.summarized_node_ids:
            already_summarized = set(summary_node.content.summarized_node_ids)

        new_messages = []
        for node in graph:
            # Filter by target IDs if specified
            if target_set and node.id not in target_set:
                continue

            # Only MESSAGE nodes
            if node.type != NodeType.MESSAGE:
                continue

            # Skip pinned nodes
            if node.metadata.pinned:
                continue

            # Skip already summarized nodes
            if node.id in already_summarized:
                continue

            # Skip nodes that are already compressed
            if node.compression_level.value >= 2:  # SUMMARIZED or higher
                continue

            # Must be newer than the summary
            node_seq = node.sequence_number or 0
            if node_seq > summary_seq:
                new_messages.append(node)

        # Sort by sequence number
        new_messages.sort(key=lambda n: n.sequence_number or 0)

        return new_messages

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute incremental summarization.

        Process:
        1. Find or create running summary node
        2. Get new messages since last summary
        3. Generate updated summary incorporating new content
        4. Update running summary node
        5. Remove processed messages

        Args:
            graph: The context graph to compress
            manifest: Recovery manifest to log operations
            target_node_ids: Optional filter to specific nodes
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics
        """
        start_time = time.perf_counter()

        # Find existing running summary
        running_summary = self._find_running_summary(graph)

        # Get new messages since the summary
        new_messages = self._get_messages_since_summary(
            graph, running_summary, target_node_ids
        )

        # Check if we have enough new messages to update
        if len(new_messages) < self._update_interval:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=len(new_messages),
                nodes_compressed=0,
                nodes_removed=0,
                nodes_created=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                is_recoverable=False,
            )

        # Calculate tokens in new messages
        new_message_tokens = sum(
            n.token_count or len(self._extract_text_from_node(n)) // 4
            for n in new_messages
        )

        # Build texts for summarization
        texts_to_summarize = []

        # Include existing summary if present
        if running_summary:
            existing_text = running_summary.content.text or ""
            if existing_text:
                texts_to_summarize.append(f"Previous context: {existing_text}")

        # Add new messages
        for msg in new_messages:
            text = self._extract_text_from_node(msg)
            if text:
                texts_to_summarize.append(text)

        # Generate updated summary
        summary_text = self._summarizer.summarize(
            texts=texts_to_summarize,
            max_tokens=self._max_summary_tokens,
            instruction="Update the running summary with new information",
        )

        # Calculate summary tokens
        summary_tokens = len(summary_text) // 4

        # Track all summarized node IDs
        all_summarized_ids = [n.id for n in new_messages]
        if running_summary and running_summary.content.summarized_node_ids:
            all_summarized_ids = (
                list(running_summary.content.summarized_node_ids) + all_summarized_ids
            )

        nodes_removed = 0
        nodes_created = 0

        if running_summary:
            # Update existing summary node
            running_summary.content.text = summary_text
            running_summary.content.summarized_node_ids = all_summarized_ids
            running_summary.content.original_tokens = (
                running_summary.content.original_tokens or 0
            ) + new_message_tokens
            running_summary.content.compressed_tokens = summary_tokens
            running_summary.token_count = summary_tokens

            # Log update operation
            manifest.log_summarize(
                node_id=new_messages[0].id,
                original_node_ids=[n.id for n in new_messages],
                summary_node_id=running_summary.id,
                original_tokens=new_message_tokens,
                summary_tokens=summary_tokens,
                method="incremental_update",
                summary_text=summary_text,
            )
        else:
            # Create new running summary node
            summary_node = self._create_summary_node(
                summary_text=summary_text,
                original_nodes=new_messages,
                summary_method="incremental",
            )
            summary_node.metadata.tags.add(self.RUNNING_SUMMARY_TAG)

            # Add to graph
            graph.add_node(summary_node, connect_temporal=True)
            nodes_created += 1
            running_summary = summary_node

            # Log creation operation
            manifest.log_summarize(
                node_id=new_messages[0].id,
                original_node_ids=[n.id for n in new_messages],
                summary_node_id=summary_node.id,
                original_tokens=new_message_tokens,
                summary_tokens=summary_tokens,
                method="incremental",
                summary_text=summary_text,
            )

        # Add SUMMARIZES edges for new messages
        for msg in new_messages:
            try:
                edge = Edge(
                    source_id=running_summary.id,
                    target_id=msg.id,
                    type=EdgeType.SUMMARIZES,
                )
                graph.add_edge(edge)
            except ValueError:
                pass

        # Remove processed messages
        for msg in new_messages:
            removed = graph.remove_node(msg.id)
            if removed:
                nodes_removed += 1

        # Calculate tokens saved
        # For updates, we need to consider the previous summary tokens
        if (
            running_summary and running_summary != summary_node
            if "summary_node" in dir()
            else False
        ):
            # This is an update - tokens saved is new messages minus growth in summary
            previous_summary_tokens = (
                running_summary.content.compressed_tokens or 0
            ) - summary_tokens
            tokens_saved = new_message_tokens - max(
                0, summary_tokens - previous_summary_tokens
            )
        else:
            tokens_saved = new_message_tokens - summary_tokens

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=new_message_tokens,
            compressed_tokens=summary_tokens,
            tokens_saved=max(0, tokens_saved),
            nodes_processed=len(new_messages),
            nodes_compressed=nodes_removed,
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=False,
        )

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved.

        Args:
            graph: The context graph
            target_node_ids: Optional filter

        Returns:
            Estimated tokens saved
        """
        running_summary = self._find_running_summary(graph)
        new_messages = self._get_messages_since_summary(
            graph, running_summary, target_node_ids
        )

        if len(new_messages) < self._update_interval:
            return 0

        new_message_tokens = sum(
            n.token_count or len(self._extract_text_from_node(n)) // 4
            for n in new_messages
        )

        # Estimate ~80% compression on new messages
        return int(new_message_tokens * 0.8)

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if incremental summarization can be applied.

        Requires enough new messages since the last summary update.

        Args:
            graph: The context graph

        Returns:
            True if summarization can be applied
        """
        running_summary = self._find_running_summary(graph)
        new_messages = self._get_messages_since_summary(graph, running_summary, None)

        return len(new_messages) >= self._update_interval
