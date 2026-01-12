"""Summarization compression strategies.

Summarization strategies create condensed representations of content.
They are irreversible - original detailed content is lost.

Strategies:
- HierarchicalSummarization: Bottom-up multi-level summarization
- TaskAwareSummarization: Task-focused summaries (to be implemented)
- IncrementalSummarization: Streaming summarization (to be implemented)
"""

from __future__ import annotations

from context_compression.strategies.summarization.hierarchical import (
    HierarchicalSummarization,
    LLMSummarizer,
    MockSummarizer,
)

__all__: list[str] = [
    "HierarchicalSummarization",
    "LLMSummarizer",
    "MockSummarizer",
]
