"""Summarization compression strategies.

Summarization strategies create condensed representations of content.
They are irreversible - original detailed content is lost.

Strategies:
- HierarchicalSummarization: Bottom-up multi-level summarization
- TaskAwareSummarization: Task-focused summaries preserving relevant context
- IncrementalSummarization: Streaming summarization with running updates

Supporting classes:
- LLMSummarizer: Protocol for LLM-based summarization backends
- SummaryResult: Result model with summary metadata
- MockLLMSummarizer: Deterministic summarizer for testing
- BaseSummarizationStrategy: Base class with shared functionality
"""

from __future__ import annotations

from context_compression.strategies.summarization.base import BaseSummarizationStrategy
from context_compression.strategies.summarization.hierarchical import (
    HierarchicalSummarization,
)
from context_compression.strategies.summarization.incremental import (
    IncrementalSummarization,
)
from context_compression.strategies.summarization.mock_summarizer import (
    LLMSummarizer,
    MockLLMSummarizer,
    SummaryResult,
)
from context_compression.strategies.summarization.task_aware import (
    TaskAwareSummarization,
)

__all__ = [
    # Protocol and models
    "LLMSummarizer",
    "SummaryResult",
    "MockLLMSummarizer",
    # Base class
    "BaseSummarizationStrategy",
    # Strategies
    "HierarchicalSummarization",
    "TaskAwareSummarization",
    "IncrementalSummarization",
]
