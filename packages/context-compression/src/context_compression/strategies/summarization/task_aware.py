"""Task-aware summarization strategy.

This strategy creates summaries that preserve task-relevant context by:
1. Extracting the current task from recent messages
2. Scoring messages by relevance to the task
3. Summarizing low-relevance messages more aggressively
4. Preserving high-relevance messages with less compression
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING
from uuid import UUID

from context_compression.strategies.summarization.base import BaseSummarizationStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.edges import Edge
from context_core.graph.nodes import ContextNode
from context_core.graph.types import EdgeType

if TYPE_CHECKING:
    from context_compression.recovery import RecoveryManifest
    from context_compression.strategies.summarization.mock_summarizer import (
        LLMSummarizer,
    )
    from context_core.graph import ContextGraph


class TaskAwareSummarization(BaseSummarizationStrategy):
    """Task-focused summarization that preserves task context.

    This strategy differs from hierarchical summarization by considering
    the relevance of each message to the current task. High-relevance
    messages are preserved or lightly summarized, while low-relevance
    messages are aggressively compressed.

    Configuration:
        summarizer: LLM backend for generating summaries
        task_context_messages: Messages to analyze for task context (default 5)
        relevance_threshold: Threshold for high-relevance (default 0.3)

    The relevance score is based on keyword overlap between messages
    and the extracted task context.
    """

    def __init__(
        self,
        summarizer: LLMSummarizer,
        task_context_messages: int = 5,
        relevance_threshold: float = 0.3,
    ) -> None:
        """Initialize the task-aware summarization strategy.

        Args:
            summarizer: LLM summarizer for generating summaries
            task_context_messages: Number of recent messages to extract task from
            relevance_threshold: Threshold for classifying as high-relevance
        """
        self._summarizer = summarizer
        self._task_context_messages = task_context_messages
        self._relevance_threshold = relevance_threshold

    @property
    def _name(self) -> str:
        return "task_aware_summarization"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.SUMMARIZATION

    @property
    def _priority(self) -> int:
        return 31  # After hierarchical

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text.

        Filters out common stop words and short words.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of lowercase keywords
        """
        # Common stop words to exclude
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
        }

        # Extract words, filter short and stop words
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())
        keywords = {w for w in words if len(w) > 2 and w not in stop_words}

        return keywords

    def _extract_task_context(self, graph: ContextGraph) -> tuple[str, set[str]]:
        """Extract current task context from recent messages.

        Analyzes the most recent messages to identify the current task
        by extracting common themes and keywords.

        Args:
            graph: The context graph

        Returns:
            Tuple of (task description, task keywords)
        """
        # Get recent messages
        all_nodes = list(graph)
        all_nodes.sort(key=lambda n: n.sequence_number or 0, reverse=True)

        recent_messages = []
        for node in all_nodes:
            if len(recent_messages) >= self._task_context_messages:
                break
            text = self._extract_text_from_node(node)
            if text.strip():
                recent_messages.append(text)

        if not recent_messages:
            return "", set()

        # Combine and extract keywords
        combined_text = " ".join(recent_messages)
        keywords = self._extract_keywords(combined_text)

        # Use first message as task description (most recent)
        task_description = recent_messages[0][:200] if recent_messages else ""

        return task_description, keywords

    def _score_relevance(
        self,
        node: ContextNode,
        task_keywords: set[str],
    ) -> float:
        """Score a node's relevance to the current task.

        Uses keyword overlap between the node's content and task keywords.

        Args:
            node: The node to score
            task_keywords: Keywords extracted from task context

        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not task_keywords:
            return 0.5  # Default when no task context

        text = self._extract_text_from_node(node)
        if not text:
            return 0.0

        node_keywords = self._extract_keywords(text)
        if not node_keywords:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(node_keywords & task_keywords)
        union = len(node_keywords | task_keywords)

        if union == 0:
            return 0.0

        return intersection / union

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute task-aware summarization.

        Process:
        1. Extract current task context from recent messages
        2. Score all messages for relevance to the task
        3. Summarize low-relevance messages more aggressively
        4. Keep high-relevance messages with less compression

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

        # Extract task context
        task_description, task_keywords = self._extract_task_context(graph)

        # Score all messages
        scored_nodes: list[tuple[ContextNode, float]] = []
        for node in message_nodes:
            relevance = self._score_relevance(node, task_keywords)
            scored_nodes.append((node, relevance))

        # Separate into high and low relevance
        low_relevance_nodes = [
            node for node, score in scored_nodes if score < self._relevance_threshold
        ]

        # Skip if no low-relevance nodes to summarize
        if not low_relevance_nodes:
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

        # Summarize low-relevance nodes in one batch
        low_relevance_tokens = sum(
            n.token_count or len(self._extract_text_from_node(n)) // 4
            for n in low_relevance_nodes
        )

        # Build text from low-relevance nodes
        low_relevance_texts = [
            self._extract_text_from_node(n) for n in low_relevance_nodes
        ]

        # Generate summary with task-aware instruction
        summary_text = self._summarizer.summarize(
            texts=low_relevance_texts,
            max_tokens=min(150, low_relevance_tokens // 5),
            instruction=f"Summarize background, focus on: {task_description[:100]}",
            preserve_entities=list(task_keywords)[:10],
        )

        # Create summary node
        summary_node = self._create_summary_node(
            summary_text=summary_text,
            original_nodes=low_relevance_nodes,
            summary_method="task_aware",
        )

        # Add summary node to graph
        graph.add_node(summary_node, connect_temporal=True)
        nodes_created += 1

        # Add SUMMARIZES edges
        for original_node in low_relevance_nodes:
            try:
                edge = Edge(
                    source_id=summary_node.id,
                    target_id=original_node.id,
                    type=EdgeType.SUMMARIZES,
                )
                graph.add_edge(edge)
            except ValueError:
                pass

        # Log operation
        summary_tokens = summary_node.token_count or 0
        manifest.log_summarize(
            node_id=low_relevance_nodes[0].id,
            original_node_ids=[n.id for n in low_relevance_nodes],
            summary_node_id=summary_node.id,
            original_tokens=low_relevance_tokens,
            summary_tokens=summary_tokens,
            method="task_aware",
            summary_text=summary_text,
        )

        # Remove low-relevance nodes
        for node in low_relevance_nodes:
            removed = graph.remove_node(node.id)
            if removed:
                nodes_removed += 1

        total_original_tokens = low_relevance_tokens
        total_compressed_tokens = summary_tokens
        total_tokens_saved = low_relevance_tokens - summary_tokens

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
        message_nodes = self._get_message_nodes(graph, target_node_ids)
        if not message_nodes:
            return 0

        # Extract task context
        _, task_keywords = self._extract_task_context(graph)

        # Count low-relevance tokens
        low_relevance_tokens = 0
        for node in message_nodes:
            relevance = self._score_relevance(node, task_keywords)
            if relevance < self._relevance_threshold:
                low_relevance_tokens += (
                    node.token_count or len(self._extract_text_from_node(node)) // 4
                )

        # Estimate ~80% compression on low-relevance content
        return int(low_relevance_tokens * 0.8)

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if task-aware summarization can be applied.

        Requires at least 2 messages with some being low-relevance.

        Args:
            graph: The context graph

        Returns:
            True if summarization can be applied
        """
        message_nodes = self._get_message_nodes(graph, None)
        if len(message_nodes) < 2:
            return False

        # Check if there are any low-relevance messages
        _, task_keywords = self._extract_task_context(graph)

        low_relevance_count = sum(
            1
            for node in message_nodes
            if self._score_relevance(node, task_keywords) < self._relevance_threshold
        )

        return low_relevance_count >= 1
