"""Mock summarizer for testing summarization strategies.

This module defines:
- LLMSummarizer: Protocol for LLM-based summarization
- SummaryResult: Result model with metadata
- MockLLMSummarizer: Deterministic summarizer for testing
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SummaryResult(BaseModel):
    """Result of summarization with metadata.

    Attributes:
        summary_text: The generated summary text
        key_entities: Entities extracted and preserved in summary
        key_decisions: Important decisions mentioned
        compression_ratio: Ratio of original to summary length
    """

    summary_text: str
    key_entities: list[str] = Field(default_factory=list)
    key_decisions: list[str] = Field(default_factory=list)
    compression_ratio: float = 1.0


@runtime_checkable
class LLMSummarizer(Protocol):
    """Protocol for LLM-based summarization.

    Implementations can use any LLM backend (OpenAI, Anthropic, local, etc.)
    to generate summaries from input texts.
    """

    def summarize(
        self,
        texts: list[str],
        max_tokens: int = 200,
        instruction: str | None = None,
        preserve_entities: list[str] | None = None,
    ) -> str:
        """Generate summary of input texts.

        Args:
            texts: List of text strings to summarize
            max_tokens: Maximum tokens in the summary
            instruction: Optional instruction for summarization focus
            preserve_entities: Entity names to preserve in the summary

        Returns:
            The summary text
        """
        ...


class MockLLMSummarizer:
    """Deterministic summarizer for testing.

    This mock summarizer creates summaries by extracting key sentences
    and truncating to fit token limits. It's designed to be predictable
    for unit testing.

    Attributes:
        compression_ratio: Target compression ratio (default 0.2)
    """

    def __init__(self, compression_ratio: float = 0.2) -> None:
        """Initialize the mock summarizer.

        Args:
            compression_ratio: Target ratio of summary to original length
        """
        self._compression_ratio = compression_ratio

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentence strings
        """
        if not text or not text.strip():
            return []

        # Split on sentence-ending punctuation followed by whitespace
        pattern = r"(?<=[.!?])\s+"
        sentences = re.split(pattern, text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses a rough approximation of 4 characters per token.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def summarize(
        self,
        texts: list[str],
        max_tokens: int = 200,
        instruction: str | None = None,
        preserve_entities: list[str] | None = None,
    ) -> str:
        """Generate summary of input texts.

        Creates a deterministic summary by:
        1. Extracting first sentence from each text
        2. Preserving sentences mentioning entities
        3. Truncating to max_tokens

        Args:
            texts: List of text strings to summarize
            max_tokens: Maximum tokens in the summary
            instruction: Optional instruction (used to add prefix)
            preserve_entities: Entity names to preserve

        Returns:
            The summary text
        """
        if not texts:
            return ""

        entities = {e.lower() for e in (preserve_entities or [])}
        collected_sentences: list[str] = []

        for text in texts:
            sentences = self._split_sentences(text)
            if not sentences:
                continue

            # Always include first sentence
            if sentences[0] not in collected_sentences:
                collected_sentences.append(sentences[0])

            # Include sentences mentioning preserved entities
            for sentence in sentences[1:]:
                if (
                    entities
                    and any(e in sentence.lower() for e in entities)
                    and sentence not in collected_sentences
                ):
                    collected_sentences.append(sentence)

        # Build summary text
        summary = " ".join(collected_sentences)

        # Truncate to max_tokens
        max_chars = max_tokens * 4
        if len(summary) > max_chars:
            # Truncate at sentence boundary if possible
            truncated = summary[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > max_chars // 2:
                summary = truncated[: last_period + 1]
            else:
                summary = truncated.rstrip() + "..."

        # Add instruction prefix if provided
        if instruction:
            prefix = f"[{instruction[:50]}] "
            if len(prefix) + len(summary) > max_chars:
                summary = summary[: max_chars - len(prefix)]
            summary = prefix + summary

        return summary

    def summarize_with_result(
        self,
        texts: list[str],
        max_tokens: int = 200,
        instruction: str | None = None,
        preserve_entities: list[str] | None = None,
    ) -> SummaryResult:
        """Generate summary with detailed result metadata.

        Args:
            texts: List of text strings to summarize
            max_tokens: Maximum tokens in the summary
            instruction: Optional instruction for summarization
            preserve_entities: Entity names to preserve

        Returns:
            SummaryResult with summary and metadata
        """
        summary_text = self.summarize(texts, max_tokens, instruction, preserve_entities)

        # Calculate compression ratio
        original_length = sum(len(t) for t in texts)
        compression_ratio = (
            len(summary_text) / original_length if original_length > 0 else 1.0
        )

        # Extract found entities
        key_entities = []
        if preserve_entities:
            summary_lower = summary_text.lower()
            for entity in preserve_entities:
                if entity.lower() in summary_lower:
                    key_entities.append(entity)

        return SummaryResult(
            summary_text=summary_text,
            key_entities=key_entities,
            key_decisions=[],  # Mock doesn't extract decisions
            compression_ratio=compression_ratio,
        )
