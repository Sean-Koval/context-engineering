"""Hierarchical summarization compression strategy.

This strategy creates multi-level summaries of context, replacing
groups of nodes with summary nodes. Summarization is irreversible -
original detailed content is lost.

Levels:
1. Message-level: Summarize long individual messages
2. Chunk-level: Summarize groups of messages
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from context_compression.recovery.operations import SummarizeOperation
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.edges import Edge
from context_core.graph.nodes import Content, ContextNode, NodeMetadata
from context_core.graph.types import CompressionLevel, EdgeType, NodeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_core.graph import ContextGraph


class LLMSummarizer(Protocol):
    """Protocol for LLM-based summarization.

    Implementations should take a list of texts and produce a summary
    within the specified token budget.
    """

    def summarize(
        self,
        texts: list[str],
        max_tokens: int,
        instruction: str | None = None,
    ) -> str:
        """Summarize texts into max_tokens.

        Args:
            texts: List of text segments to summarize
            max_tokens: Maximum tokens for the summary
            instruction: Optional instruction for guiding summarization

        Returns:
            The summary text
        """
        ...


class MockSummarizer:
    """Mock summarizer for testing that creates deterministic summaries.

    Produces predictable summaries based on input characteristics,
    useful for testing without requiring an actual LLM.
    """

    def summarize(
        self,
        texts: list[str],
        max_tokens: int,
        instruction: str | None = None,
    ) -> str:
        """Create a predictable summary based on input.

        Args:
            texts: List of text segments to summarize
            max_tokens: Maximum tokens for the summary
            instruction: Optional instruction (ignored in mock)

        Returns:
            A deterministic summary string
        """
        total_chars = sum(len(t) for t in texts)
        return f"[Summary of {len(texts)} messages, {total_chars} chars]"


class HierarchicalSummarization(BaseCompressionStrategy):
    """Create hierarchical summaries of context.

    This strategy groups messages into chunks and creates summary nodes
    that replace the original content. It operates at two levels:

    1. Message-level: Long individual messages (above message_threshold)
       are summarized individually.
    2. Chunk-level: Groups of messages (based on chunk_size or chunk_threshold)
       are summarized together.

    The strategy:
    - Creates SUMMARY nodes with the summarized content
    - Adds SUMMARIZES edges from summary to original nodes
    - Removes original nodes after summarization
    - Logs SummarizeOperation to the recovery manifest

    This is SUMMARIZATION tier - irreversible, original content is lost.

    Configuration:
        summarizer: LLM summarizer implementing the LLMSummarizer protocol
        chunk_size: Number of messages per chunk (default 10)
        message_threshold: Token count to trigger message-level summary (default 500)
        chunk_threshold: Token count to trigger chunk-level summary (default 2000)
        preserve_recent: Number of recent messages to never summarize (default 5)

    Example:
        >>> from context_compression.strategies.summarization import (
        ...     HierarchicalSummarization, MockSummarizer
        ... )
        >>> summarizer = MockSummarizer()
        >>> strategy = HierarchicalSummarization(
        ...     summarizer=summarizer,
        ...     chunk_size=10,
        ...     preserve_recent=5,
        ... )
        >>> result = strategy.compress(graph, manifest)
    """

    def __init__(
        self,
        summarizer: LLMSummarizer,
        chunk_size: int = 10,
        message_threshold: int = 500,
        chunk_threshold: int = 2000,
        preserve_recent: int = 5,
    ) -> None:
        """Initialize the hierarchical summarization strategy.

        Args:
            summarizer: LLM summarizer for generating summaries
            chunk_size: Maximum number of messages per chunk
            message_threshold: Token threshold for message-level summarization
            chunk_threshold: Token threshold for chunk-level summarization
            preserve_recent: Number of most recent messages to never summarize
        """
        self._summarizer = summarizer
        self._chunk_size = chunk_size
        self._message_threshold = message_threshold
        self._chunk_threshold = chunk_threshold
        self._preserve_recent = preserve_recent

    @property
    def _name(self) -> str:
        return "hierarchical_summarization"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.SUMMARIZATION

    @property
    def _priority(self) -> int:
        return 10

    def _get_node_text(self, node: ContextNode) -> str:
        """Extract text content from a node.

        Args:
            node: The context node to extract text from

        Returns:
            Extracted text, or empty string if no content
        """
        content = node.content

        # Message text
        if content.text:
            return content.text

        # Tool calls: combine name and args
        if content.tool_name:
            args_str = str(content.tool_args) if content.tool_args else ""
            return f"{content.tool_name}: {args_str}"

        # Tool output
        if content.tool_output is not None:
            return str(content.tool_output)[:1000]

        # Artifact data
        if content.artifact_data is not None:
            return str(content.artifact_data)[:1000]

        return ""

    def _get_eligible_nodes(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> list[ContextNode]:
        """Get nodes eligible for summarization.

        Excludes:
        - Pinned nodes
        - Already summarized nodes (SUMMARIZED or EVICTED)
        - SUMMARY nodes (don't summarize summaries)
        - SYSTEM nodes (important for task context)
        - Recent nodes (based on preserve_recent)

        Args:
            graph: The context graph
            target_node_ids: Optional list of specific node IDs to target

        Returns:
            List of eligible nodes sorted by sequence
        """
        # Get all nodes sorted by sequence
        nodes = sorted(graph, key=lambda n: n.sequence_number or 0)

        if not nodes:
            return []

        # Determine protected node IDs (most recent N)
        protected_ids: set[UUID] = set()
        if self._preserve_recent > 0:
            if len(nodes) <= self._preserve_recent:
                # All nodes are protected
                protected_ids = {n.id for n in nodes}
            else:
                # Protect the most recent N nodes
                protected_ids = {n.id for n in nodes[-self._preserve_recent :]}

        eligible: list[ContextNode] = []

        for node in nodes:
            # Skip if targeting specific nodes and not in target list
            if target_node_ids is not None and node.id not in target_node_ids:
                continue

            # Skip protected recent nodes
            if node.id in protected_ids:
                continue

            # Skip pinned nodes
            if node.metadata.pinned:
                continue

            # Skip already compressed nodes (SUMMARIZED or EVICTED)
            if node.compression_level >= CompressionLevel.SUMMARIZED:
                continue

            # Skip SUMMARY nodes (don't summarize summaries)
            if node.type == NodeType.SUMMARY:
                continue

            # Skip SYSTEM nodes (they define the task)
            if node.type == NodeType.SYSTEM:
                continue

            eligible.append(node)

        return eligible

    def _group_into_chunks(
        self,
        nodes: list[ContextNode],
    ) -> list[list[ContextNode]]:
        """Group nodes into chunks for summarization.

        Groups are formed based on:
        - Maximum chunk_size messages per group
        - Maximum chunk_threshold tokens per group

        Args:
            nodes: List of eligible nodes to group

        Returns:
            List of node chunks (each chunk is a list of nodes)
        """
        if not nodes:
            return []

        chunks: list[list[ContextNode]] = []
        current_chunk: list[ContextNode] = []
        current_tokens = 0

        for node in nodes:
            node_tokens = node.token_count or 0
            current_chunk.append(node)
            current_tokens += node_tokens

            # Close chunk if size or token limit reached
            if (
                len(current_chunk) >= self._chunk_size
                or current_tokens >= self._chunk_threshold
            ):
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0

        # Don't forget leftover nodes
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_summary_node(
        self,
        chunk: list[ContextNode],
        summary_text: str,
    ) -> ContextNode:
        """Create a summary node for a chunk of nodes.

        Args:
            chunk: List of nodes being summarized
            summary_text: The summary text

        Returns:
            A new SUMMARY node
        """
        # Compute max importance from chunk
        max_importance = max(n.compute_importance() for n in chunk)

        # Collect all tags
        all_tags: set[str] = {"hierarchical_summary"}
        for node in chunk:
            all_tags.update(node.metadata.tags)

        # Collect all entities
        all_entities: list[str] = []
        seen_entities: set[str] = set()
        for node in chunk:
            for entity_id in node.metadata.entities:
                if entity_id not in seen_entities:
                    all_entities.append(entity_id)
                    seen_entities.add(entity_id)

        # Use creation time of first node in chunk
        created_at = chunk[0].metadata.created_at

        # Estimate token count for summary
        summary_tokens = len(summary_text) // 4 + 1

        summary_node = ContextNode(
            type=NodeType.SUMMARY,
            content=Content(
                text=summary_text,
                summarized_node_ids=[n.id for n in chunk],
                summary_method="hierarchical",
            ),
            metadata=NodeMetadata(
                importance=max_importance,
                tags=all_tags,
                entities=all_entities,
                created_at=created_at,
            ),
            compression_level=CompressionLevel.SUMMARIZED,
            token_count=summary_tokens,
        )

        return summary_node

    def _format_node_for_summary(self, node: ContextNode) -> str:
        """Format a node's content for the summarizer.

        Args:
            node: The node to format

        Returns:
            Formatted string representation
        """
        role = "unknown"
        if node.type == NodeType.MESSAGE and node.content.role:
            role = node.content.role.value
        elif node.type == NodeType.TOOL_CALL:
            role = "tool_call"
        elif node.type == NodeType.TOOL_RESULT:
            role = "tool_result"

        text = self._get_node_text(node)
        return f"{role}: {text}"

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved by summarization.

        Args:
            graph: The context graph to analyze
            target_node_ids: Optional list of node IDs to target

        Returns:
            Estimated number of tokens that would be saved
        """
        eligible = self._get_eligible_nodes(graph, target_node_ids)
        chunks = self._group_into_chunks(eligible)

        total_savings = 0

        for chunk in chunks:
            chunk_tokens = sum(n.token_count or 0 for n in chunk)
            # Estimate summary size: ~20% of original, minimum 50 tokens
            estimated_summary_tokens = max(50, chunk_tokens // 5)
            savings = max(0, chunk_tokens - estimated_summary_tokens)
            total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute hierarchical summarization.

        Args:
            graph: The context graph to compress (modified in place)
            manifest: Recovery manifest to log operations
            target_node_ids: Optional list of node IDs to target
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        start_time = time.perf_counter()

        # Get eligible nodes and group into chunks
        eligible = self._get_eligible_nodes(graph, target_node_ids)
        chunks = self._group_into_chunks(eligible)

        # Track metrics
        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_processed = 0
        nodes_removed = 0
        nodes_created = 0

        # Process each chunk
        for chunk in chunks:
            # Check if we've reached target
            if target_tokens is not None and tokens_saved >= target_tokens:
                break

            chunk_tokens = sum(n.token_count or 0 for n in chunk)
            original_tokens += chunk_tokens
            nodes_processed += len(chunk)

            # Build texts for summarizer
            chunk_texts = [self._format_node_for_summary(node) for node in chunk]

            # Generate summary (target ~20% of original, min 50 tokens)
            max_summary_tokens = max(50, chunk_tokens // 5)
            instruction = (
                "Summarize this conversation chunk, preserving key decisions, "
                "actions, and entity mentions."
            )
            summary_text = self._summarizer.summarize(
                texts=chunk_texts,
                max_tokens=max_summary_tokens,
                instruction=instruction,
            )

            # Create summary node
            summary_node = self._create_summary_node(chunk, summary_text)
            summary_tokens = summary_node.token_count or 0

            # Add summary node to graph (without temporal edges)
            graph.add_node(summary_node, connect_temporal=False)
            nodes_created += 1

            # Create SUMMARIZES edges from summary to originals
            for node in chunk:
                edge = Edge(
                    source_id=summary_node.id,
                    target_id=node.id,
                    type=EdgeType.SUMMARIZES,
                )
                graph.add_edge(edge)

            # Log operation to manifest
            manifest.log_operation(
                SummarizeOperation(
                    node_id=chunk[0].id,  # Primary node for the operation
                    original_node_ids=[n.id for n in chunk],
                    summary_node_id=summary_node.id,
                    original_tokens=chunk_tokens,
                    summary_tokens=summary_tokens,
                    method="hierarchical",
                    summary_text=summary_text,
                )
            )

            # Remove original nodes from graph
            for node in chunk:
                graph.remove_node(node.id)
                nodes_removed += 1

            compressed_tokens += summary_tokens
            tokens_saved += chunk_tokens - summary_tokens

        duration_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=nodes_processed,
            nodes_compressed=0,  # We don't modify in place, we replace
            nodes_removed=nodes_removed,
            nodes_created=nodes_created,
            duration_ms=duration_ms,
            is_recoverable=False,  # Summarization is not recoverable
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if hierarchical summarization can be applied.

        Returns True if there are enough eligible nodes to form at least
        one chunk of size >= chunk_size.

        Args:
            graph: The context graph to check

        Returns:
            True if the strategy can be applied
        """
        eligible = self._get_eligible_nodes(graph, None)
        return len(eligible) >= self._chunk_size
