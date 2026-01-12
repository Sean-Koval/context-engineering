"""Task relevance compression strategy.

This strategy identifies content not relevant to the current task and
compresses or summarizes it, preserving task-focused context while
reducing token usage.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from context_compression.recovery import RecoveryManifest
from context_compression.recovery.operations import TaskRelevanceOperation
from context_compression.strategies.base import BaseCompressionStrategy
from context_compression.types import CompressionResult, CompressionTier
from context_core.graph.types import CompressionLevel, NodeType, Role

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from context_core.graph import ContextGraph, ContextNode
    from context_core.semantic import SemanticIndex


class TaskRelevanceCompression(BaseCompressionStrategy):
    """Compresses content not relevant to the current task.

    Uses semantic similarity to identify off-task content and compresses
    or summarizes nodes that fall below a relevance threshold. Combines
    semantic similarity, recency, and importance scoring to make
    intelligent compression decisions.

    This is COMPACTION tier - some information may be lost but the overall
    structure and key context is preserved. Off-task content is summarized
    rather than removed entirely.

    Configuration:
        semantic_index: SemanticIndex for computing embeddings and similarity
        relevance_threshold: Minimum relevance score to keep (default 0.3)
        min_age_to_compress: Don't compress the N most recent nodes (default 5)
        recency_weight: Weight for recency in scoring (default 0.3)
        importance_weight: Weight for importance in scoring (default 0.2)
        semantic_weight: Weight for semantic similarity (default 0.5)

    The strategy is partially reversible - original content is stored
    in recovery operations but semantic context may be lost.

    Example:
        >>> from context_core.semantic import SemanticIndex, MockEmbeddingModel
        >>> model = MockEmbeddingModel(dimension=64)
        >>> index = SemanticIndex(model)
        >>> strategy = TaskRelevanceCompression(
        ...     semantic_index=index,
        ...     relevance_threshold=0.3,
        ... )
        >>> result = strategy.compress(graph, manifest)
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        relevance_threshold: float = 0.3,
        min_age_to_compress: int = 5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2,
        semantic_weight: float = 0.5,
    ) -> None:
        """Initialize the strategy.

        Args:
            semantic_index: SemanticIndex for computing embeddings and similarity.
            relevance_threshold: Nodes below this relevance score are compressed.
                Range 0.0-1.0, default 0.3.
            min_age_to_compress: Don't compress the N most recent nodes.
                These are preserved regardless of relevance. Default 5.
            recency_weight: Weight for recency factor in scoring.
                Higher values favor newer content. Default 0.3.
            importance_weight: Weight for node importance in scoring.
                Uses node's compute_importance(). Default 0.2.
            semantic_weight: Weight for semantic similarity in scoring.
                Based on cosine similarity to task context. Default 0.5.
        """
        self._semantic_index = semantic_index
        self._relevance_threshold = relevance_threshold
        self._min_age_to_compress = min_age_to_compress
        self._recency_weight = recency_weight
        self._importance_weight = importance_weight
        self._semantic_weight = semantic_weight

    @property
    def _name(self) -> str:
        return "task_relevance_compression"

    @property
    def _tier(self) -> CompressionTier:
        return CompressionTier.COMPACTION

    @property
    def _priority(self) -> int:
        return 30  # Run after schema compression (priority 10)

    def _extract_task_context(self, graph: ContextGraph) -> str:
        """Extract the current task from recent context.

        Looks for task-defining content in:
        1. System prompt content (highest priority)
        2. Recent user messages with instructions
        3. Messages with high importance scores

        Args:
            graph: The context graph to analyze

        Returns:
            A string representing the current task context.
            Returns empty string if no task context found.
        """
        task_parts: list[str] = []

        # Collect all nodes sorted by sequence
        nodes = list(graph)
        if not nodes:
            return ""

        # 1. Get system prompt content (always most relevant for task)
        for node in nodes:
            if node.type == NodeType.SYSTEM and node.content.text:
                task_parts.append(node.content.text)

        # 2. Get recent user messages (last 3-5)
        user_messages = [
            node
            for node in nodes
            if node.type == NodeType.MESSAGE
            and node.content.role == Role.USER
            and node.content.text
        ]
        # Take last 3 user messages
        for node in user_messages[-3:]:
            if node.content.text:
                task_parts.append(node.content.text)

        # 3. Get high-importance messages
        important_nodes = sorted(
            [n for n in nodes if n.type == NodeType.MESSAGE and n.content.text],
            key=lambda n: n.compute_importance(),
            reverse=True,
        )
        for node in important_nodes[:2]:
            if node.content.text and node.content.text not in task_parts:
                task_parts.append(node.content.text)

        return " ".join(task_parts)

    def _get_task_embedding(self, task_context: str) -> NDArray[np.float32] | None:
        """Get embedding for the task context.

        Args:
            task_context: The extracted task context string

        Returns:
            Embedding vector or None if context is empty
        """
        if not task_context.strip():
            return None

        embeddings = self._semantic_index.embedding_model.embed([task_context])
        return embeddings[0] if len(embeddings) > 0 else None

    def _get_node_text(self, node: ContextNode) -> str | None:
        """Extract text content from a node for embedding.

        Args:
            node: The context node to extract text from

        Returns:
            Extracted text, or None if no indexable content
        """
        content = node.content

        # Try text content first
        if content.text:
            return content.text

        # Tool calls: combine name and args
        if content.tool_name:
            args_str = str(content.tool_args) if content.tool_args else ""
            return f"{content.tool_name}: {args_str}"

        # Tool output: stringify and truncate
        if content.tool_output is not None:
            return str(content.tool_output)[:1000]

        # Artifact data: stringify and truncate
        if content.artifact_data is not None:
            return str(content.artifact_data)[:1000]

        return None

    def _compute_semantic_similarity(
        self,
        node: ContextNode,
        task_embedding: NDArray[np.float32],
    ) -> float:
        """Compute semantic similarity between node and task.

        Args:
            node: The node to score
            task_embedding: Embedding of the task context

        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        node_text = self._get_node_text(node)
        if not node_text:
            return 0.0

        # Get node embedding
        node_embedding = self._semantic_index.embedding_model.embed([node_text])[0]

        # Compute cosine similarity
        dot_product = np.dot(node_embedding, task_embedding)
        norm_node = np.linalg.norm(node_embedding)
        norm_task = np.linalg.norm(task_embedding)

        if norm_node == 0 or norm_task == 0:
            return 0.0

        similarity = dot_product / (norm_node * norm_task)

        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return float((similarity + 1) / 2)

    def _compute_relevance_score(
        self,
        node: ContextNode,
        task_embedding: NDArray[np.float32] | None,
        max_sequence: int,
    ) -> float:
        """Compute relevance score for a node.

        Combines:
        - Semantic similarity to task (semantic_weight)
        - Recency factor (recency_weight)
        - Importance score (importance_weight)

        Args:
            node: The node to score
            task_embedding: Embedding of the task context (None = skip semantic)
            max_sequence: Maximum sequence number in graph (for recency calc)

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Semantic similarity component
        if task_embedding is not None:
            semantic_score = self._compute_semantic_similarity(node, task_embedding)
        else:
            # If no task context, use neutral score
            semantic_score = 0.5

        # Recency component (newer = higher score)
        if max_sequence > 0 and node.sequence_number is not None:
            recency_score = node.sequence_number / max_sequence
        else:
            recency_score = 0.5

        # Importance component (from node's compute_importance)
        importance_score = node.compute_importance()

        # Combine with weights
        relevance = (
            self._semantic_weight * semantic_score
            + self._recency_weight * recency_score
            + self._importance_weight * importance_score
        )

        return min(1.0, max(0.0, relevance))

    def _is_eligible(self, node: ContextNode, protected_ids: set[UUID]) -> bool:
        """Check if a node is eligible for task relevance compression.

        Args:
            node: The node to check
            protected_ids: Set of node IDs that should be protected

        Returns:
            True if the node can be compressed
        """
        # Don't compress protected nodes (recent ones)
        if node.id in protected_ids:
            return False

        # Don't compress pinned nodes
        if node.metadata.pinned:
            return False

        # Don't compress already compressed nodes
        if node.compression_level != CompressionLevel.FULL:
            return False

        # Don't compress system messages (they define the task)
        if node.type == NodeType.SYSTEM:
            return False

        # Must have content to compress
        return self._get_node_text(node) is not None

    def _create_compressed_content(
        self,
        node: ContextNode,
        relevance_score: float,
    ) -> str:
        """Create a compressed summary of node content.

        Args:
            node: The node to compress
            relevance_score: The node's relevance score

        Returns:
            Compressed content string
        """
        original = self._get_node_text(node) or ""

        # Create a brief summary based on node type
        if node.type == NodeType.TOOL_CALL:
            tool_name = node.content.tool_name or "unknown_tool"
            return f"[Off-task tool call: {tool_name}]"

        if node.type == NodeType.TOOL_RESULT:
            # Keep first part of output as preview
            preview = original[:100].replace("\n", " ").strip()
            if len(original) > 100:
                preview += "..."
            return f"[Off-task result: {preview}]"

        if node.type == NodeType.MESSAGE:
            role = node.content.role.value if node.content.role else "unknown"
            preview = original[:100].replace("\n", " ").strip()
            if len(original) > 100:
                preview += "..."
            return f"[Off-task {role} message: {preview}]"

        # Default: truncate content
        preview = original[:100].replace("\n", " ").strip()
        if len(original) > 100:
            preview += "..."
        return f"[Off-task content: {preview}]"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation of 4 characters per token.

        Args:
            text: The text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4 + 1

    def _estimate_savings_impl(
        self,
        graph: ContextGraph,
        target_node_ids: list[UUID] | None,
    ) -> int:
        """Estimate tokens that would be saved.

        Args:
            graph: The context graph to analyze
            target_node_ids: Optional list of node IDs to target

        Returns:
            Estimated number of tokens that would be saved
        """
        # Get task context and embedding
        task_context = self._extract_task_context(graph)
        task_embedding = self._get_task_embedding(task_context)

        # Get max sequence for recency calculation
        nodes = list(graph)
        if not nodes:
            return 0

        max_sequence = max((n.sequence_number or 0) for n in nodes)

        # Determine protected nodes (most recent N)
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number or 0, reverse=True)
        protected_ids = {n.id for n in sorted_nodes[: self._min_age_to_compress]}

        target_ids = set(target_node_ids) if target_node_ids else None

        total_savings = 0

        for node in nodes:
            # Skip if not targeted
            if target_ids and node.id not in target_ids:
                continue

            # Skip if not eligible
            if not self._is_eligible(node, protected_ids):
                continue

            # Compute relevance score
            score = self._compute_relevance_score(node, task_embedding, max_sequence)

            # If below threshold, estimate savings
            if score < self._relevance_threshold:
                original_text = self._get_node_text(node) or ""
                compressed = self._create_compressed_content(node, score)

                original_tokens = self._estimate_tokens(original_text)
                compressed_tokens = self._estimate_tokens(compressed)
                savings = max(0, original_tokens - compressed_tokens)
                total_savings += savings

        return total_savings

    def _compress_impl(
        self,
        graph: ContextGraph,
        manifest: RecoveryManifest,
        target_node_ids: list[UUID] | None,
        target_tokens: int | None,
    ) -> CompressionResult:
        """Execute task relevance compression.

        Args:
            graph: The context graph to compress (modified in place)
            manifest: Recovery manifest to log operations
            target_node_ids: Optional list of node IDs to target
            target_tokens: Stop when this many tokens saved

        Returns:
            CompressionResult with metrics about the operation
        """
        start_time = time.perf_counter()

        # Get task context and embedding
        task_context = self._extract_task_context(graph)
        task_embedding = self._get_task_embedding(task_context)
        task_preview = task_context[:100] if task_context else ""

        # Get all nodes
        nodes = list(graph)
        if not nodes:
            return CompressionResult(
                success=True,
                strategy_name=self.name,
                tier=self.tier,
                original_tokens=0,
                compressed_tokens=0,
                tokens_saved=0,
                nodes_processed=0,
                is_recoverable=False,
            )

        max_sequence = max((n.sequence_number or 0) for n in nodes)

        # Determine protected nodes (most recent N)
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number or 0, reverse=True)
        protected_ids = {n.id for n in sorted_nodes[: self._min_age_to_compress]}

        target_ids = set(target_node_ids) if target_node_ids else None

        # Track metrics
        original_tokens = 0
        compressed_tokens = 0
        tokens_saved = 0
        nodes_compressed = 0

        # Score all eligible nodes
        scored_nodes: list[tuple[ContextNode, float]] = []
        for node in nodes:
            if target_ids and node.id not in target_ids:
                continue
            if not self._is_eligible(node, protected_ids):
                continue

            score = self._compute_relevance_score(node, task_embedding, max_sequence)
            if score < self._relevance_threshold:
                scored_nodes.append((node, score))

        # Sort by relevance (lowest first) to compress least relevant first
        scored_nodes.sort(key=lambda x: x[1])

        for node, score in scored_nodes:
            # Check target limit
            if target_tokens and tokens_saved >= target_tokens:
                break

            # Get original content
            original_text = self._get_node_text(node) or ""
            node_original_tokens = node.token_count or self._estimate_tokens(
                original_text
            )

            # Create compressed content
            compressed_content = self._create_compressed_content(node, score)
            node_compressed_tokens = self._estimate_tokens(compressed_content)

            # Calculate savings
            node_savings = max(0, node_original_tokens - node_compressed_tokens)
            if node_savings <= 0:
                continue

            # Update tokens
            original_tokens += node_original_tokens
            compressed_tokens += node_compressed_tokens
            tokens_saved += node_savings

            # Update node content based on type
            if node.type == NodeType.MESSAGE:
                node.content.text = compressed_content
            elif node.type == NodeType.TOOL_CALL:
                # Keep tool name but summarize args
                node.content.tool_args = {"_compressed": compressed_content}
            elif node.type == NodeType.TOOL_RESULT:
                node.content.tool_output = compressed_content
            else:
                # For other types, try to update text or artifact_data
                if node.content.text is not None:
                    node.content.text = compressed_content
                elif node.content.artifact_data is not None:
                    node.content.artifact_data = compressed_content

            # Update compression state
            node.compression_level = CompressionLevel.COMPACTED
            node.content.original_tokens = node_original_tokens
            node.content.compressed_tokens = node_compressed_tokens
            node.token_count = node_compressed_tokens

            # Log operation to manifest
            manifest.log_operation(
                TaskRelevanceOperation(
                    node_id=node.id,
                    relevance_score=score,
                    task_context_preview=task_preview,
                    original_content=original_text,
                    compressed_content=compressed_content,
                    original_tokens=node_original_tokens,
                    compressed_tokens=node_compressed_tokens,
                )
            )

            nodes_compressed += 1

        duration_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            success=True,
            strategy_name=self.name,
            tier=self.tier,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            nodes_processed=len(nodes),
            nodes_compressed=nodes_compressed,
            nodes_removed=0,
            nodes_created=0,
            duration_ms=duration_ms,
            is_recoverable=False,  # Compaction is partially recoverable
        )

    def _can_apply_impl(self, graph: ContextGraph) -> bool:
        """Check if task relevance compression can be applied.

        Args:
            graph: The context graph to check

        Returns:
            True if there are nodes that could be compressed
        """
        nodes = list(graph)
        if not nodes:
            return False

        # Need at least min_age_to_compress + 1 nodes to have anything to compress
        if len(nodes) <= self._min_age_to_compress:
            return False

        # Get task context
        task_context = self._extract_task_context(graph)
        task_embedding = self._get_task_embedding(task_context)

        max_sequence = max((n.sequence_number or 0) for n in nodes)

        # Determine protected nodes
        sorted_nodes = sorted(nodes, key=lambda n: n.sequence_number or 0, reverse=True)
        protected_ids = {n.id for n in sorted_nodes[: self._min_age_to_compress]}

        # Check if any eligible node is below threshold
        for node in nodes:
            if not self._is_eligible(node, protected_ids):
                continue

            score = self._compute_relevance_score(node, task_embedding, max_sequence)
            if score < self._relevance_threshold:
                return True

        return False
