"""Hierarchical summarization strategy.

This strategy creates bottom-up multi-level summaries by:
1. Grouping messages into chunks
2. Summarizing each chunk
3. Optionally creating higher-level summaries of summaries
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.strategies.summarization.base import BaseSummarizationStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.edges import Edge
from context_core.graph.types import EdgeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_compression.strategies.summarization.mock_summarizer import (
        LLMSummarizer,
    )
    from context_core.graph import ContextGraph


class HierarchicalSummarization(BaseSummarizationStrategy):
    """Bottom-up summarization creating multi-level summaries.

    This strategy groups messages into chunks and summarizes each chunk
    independently. It preserves the most recent chunks to maintain
    context continuity.

    Configuration:
        summarizer: LLM backend for generating summaries
        chunk_size: Maximum messages per chunk (default 10)
        chunk_token_threshold: Maximum tokens per chunk (default 2000)
        preserve_recent_chunks: Number of recent chunks to keep (default 1)

    The strategy creates SUMMARY nodes with SUMMARIZES edges pointing
    to the original messages, then removes the originals.
    """

    def __init__(
        self,
        summarizer: LLMSummarizer,
        chunk_size: int = 10,
        chunk_token_threshold: int = 2000,
        preserve_recent_chunks: int = 1,
    ) -> None:
        """Initialize the hierarchical summarization strategy.

        Args:
            summarizer: LLM summarizer for generating summaries
            chunk_size: Maximum messages per chunk
            chunk_token_threshold: Maximum tokens per chunk
            preserve_recent_chunks: Number of recent chunks to preserve
        """
        self._summarizer = summarizer
        self._chunk_size = chunk_size
        self._chunk_token_threshold = chunk_token_threshold
        self._preserve_recent_chunks = preserve_recent_chunks

    @property
    def _name(self) -> str:
        return "hierarchical_summarization"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.SUMMARIZATION

    @property
    def _priority(self) -> int:
        return 30

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute hierarchical summarization.

        Process:
        1. Get MESSAGE nodes ordered by sequence
        2. Group into chunks by size OR token threshold
        3. For each chunk (except recent ones):
           a. Build chunk text from messages
           b. Call summarizer
           c. Create SUMMARY node with SUMMARIZES edges
           d. Log SummarizeOperation to manifest
           e. Remove original nodes

        Args:
            graph: The context graph to compress
            manifest: Recovery manifest to log operations
            target_node_ids: Optional filter to specific nodes
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics
        """
        start_time = time.perf_counter()

        # Get eligible MESSAGE nodes
        message_nodes = self._get_message_nodes(graph, target_node_ids)

        if not message_nodes:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=0,
                nodes_compressed=0,
                nodes_removed=0,
                nodes_created=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                is_recoverable=False,
            )

        # Chunk the messages
        chunks = self._chunk_nodes(
            message_nodes,
            chunk_size=self._chunk_size,
            token_threshold=self._chunk_token_threshold,
        )

        # Determine which chunks to summarize (exclude recent ones)
        chunks_to_summarize = chunks[: len(chunks) - self._preserve_recent_chunks]

        if not chunks_to_summarize:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=len(message_nodes),
                nodes_compressed=0,
                nodes_removed=0,
                nodes_created=0,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                is_recoverable=False,
            )

        total_original_tokens = 0
        total_compressed_tokens = 0
        total_tokens_saved = 0
        nodes_removed = 0
        nodes_created = 0

        for chunk in chunks_to_summarize:
            # Check target limit
            if target_tokens and total_tokens_saved >= target_tokens:
                break

            # Calculate chunk tokens
            chunk_tokens = sum(
                n.token_count or len(self._extract_text_from_node(n)) // 4
                for n in chunk
            )

            # Build text from chunk
            chunk_texts = [self._extract_text_from_node(n) for n in chunk]

            # Generate summary
            summary_text = self._summarizer.summarize(
                texts=chunk_texts,
                max_tokens=min(200, chunk_tokens // 3),
                instruction="Summarize the conversation",
            )

            # Create summary node
            summary_node = self._create_summary_node(
                summary_text=summary_text,
                original_nodes=chunk,
                summary_method="hierarchical",
            )

            # Add summary node to graph
            graph.add_node(summary_node, connect_temporal=True)
            nodes_created += 1

            # Add SUMMARIZES edges from summary to original nodes
            for original_node in chunk:
                try:
                    edge = Edge(
                        source_id=summary_node.id,
                        target_id=original_node.id,
                        type=EdgeType.SUMMARIZES,
                    )
                    graph.add_edge(edge)
                except ValueError:
                    # Node may already be removed
                    pass

            # Log operation to manifest
            summary_tokens = summary_node.token_count or 0
            manifest.log_summarize(
                node_id=chunk[0].id,  # Use first node as primary
                original_node_ids=[n.id for n in chunk],
                summary_node_id=summary_node.id,
                original_tokens=chunk_tokens,
                summary_tokens=summary_tokens,
                method="hierarchical",
                summary_text=summary_text,
            )

            # Remove original nodes
            for node in chunk:
                removed = graph.remove_node(node.id)
                if removed:
                    nodes_removed += 1

            # Update totals
            total_original_tokens += chunk_tokens
            total_compressed_tokens += summary_tokens
            tokens_saved = chunk_tokens - summary_tokens
            total_tokens_saved += tokens_saved

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=total_original_tokens,
            compressed_tokens=total_compressed_tokens,
            tokens_saved=total_tokens_saved,
            nodes_processed=len(message_nodes),
            nodes_compressed=nodes_removed,
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=(time.perf_counter() - start_time) * 1000,
            is_recoverable=False,  # Summarization is not recoverable
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
        message_nodes = self._get_message_nodes(graph, target_node_ids)
        if not message_nodes:
            return 0

        chunks = self._chunk_nodes(
            message_nodes,
            chunk_size=self._chunk_size,
            token_threshold=self._chunk_token_threshold,
        )

        chunks_to_summarize = chunks[: len(chunks) - self._preserve_recent_chunks]

        total_tokens = 0
        for chunk in chunks_to_summarize:
            chunk_tokens = sum(
                n.token_count or len(self._extract_text_from_node(n)) // 4
                for n in chunk
            )
            total_tokens += chunk_tokens

        # Estimate ~80% compression
        return int(total_tokens * 0.8)

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if hierarchical summarization can be applied.

        Requires enough messages to form at least one chunk to summarize
        while preserving recent chunks.

        Args:
            graph: The context graph

        Returns:
            True if summarization can be applied
        """
        message_nodes = self._get_message_nodes(graph, None)
        if len(message_nodes) < 2:
            return False

        chunks = self._chunk_nodes(
            message_nodes,
            chunk_size=self._chunk_size,
            token_threshold=self._chunk_token_threshold,
        )

        # Need at least one chunk to summarize after preserving recent
        return len(chunks) > self._preserve_recent_chunks
