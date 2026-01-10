"""Tokenizer implementations for model-agnostic token counting.

This module provides:
- Tokenizer: Protocol for tokenizer implementations
- TiktokenTokenizer: OpenAI tiktoken-based implementation
- EstimateTokenizer: Fallback character-based estimation
- get_tokenizer: Factory function for getting appropriate tokenizer
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for tokenizer implementations.

    All tokenizers must implement these methods for counting
    and truncating tokens. Implementations should be model-aware
    where possible for accurate counting.
    """

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: The text to tokenize and count.

        Returns:
            The number of tokens in the text.
        """
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token limit.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            The truncated text, guaranteed to be <= max_tokens.
        """
        ...


class TiktokenTokenizer:
    """Tokenizer using tiktoken for OpenAI models.

    This is the most accurate tokenizer for GPT models.
    Requires the tiktoken optional dependency to be installed.

    Example:
        >>> tokenizer = TiktokenTokenizer("gpt-4")
        >>> tokenizer.count_tokens("Hello, world!")
        4
    """

    def __init__(self, model: str = "gpt-4", cache_size: int = 10000) -> None:
        """Initialize tokenizer for a specific model.

        Args:
            model: The OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo").
            cache_size: Maximum number of cached token counts (default 10000).

        Raises:
            ImportError: If tiktoken is not installed.
            KeyError: If the model is not recognized.
        """
        import tiktoken

        self._model = model
        self._encoding = tiktoken.encoding_for_model(model)
        self._cache: dict[str, int] = {}
        self._cache_size = cache_size

    @property
    def model(self) -> str:
        """The model this tokenizer is configured for."""
        return self._model

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Results are cached for performance. Cache is bounded to prevent
        unbounded memory growth.

        Args:
            text: The text to tokenize.

        Returns:
            The number of tokens.
        """
        if text in self._cache:
            return self._cache[text]

        count = len(self._encoding.encode(text))

        # Simple LRU-ish behavior: clear half when full
        if len(self._cache) >= self._cache_size:
            # Remove oldest half of entries
            keys_to_remove = list(self._cache.keys())[: self._cache_size // 2]
            for key in keys_to_remove:
                del self._cache[key]

        self._cache[text] = count
        return count

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens.

        Returns:
            The truncated text.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])


class EstimateTokenizer:
    """Fallback tokenizer using character-based estimation.

    Uses a configurable characters-per-token ratio for estimation.
    Less accurate than TiktokenTokenizer but has no dependencies.

    The default ratio of 4.0 characters per token is a reasonable
    approximation for English text with GPT-style tokenizers.

    Example:
        >>> tokenizer = EstimateTokenizer(chars_per_token=4.0)
        >>> tokenizer.count_tokens("Hello, world!")  # 13 chars -> 3 tokens
        3
    """

    def __init__(self, chars_per_token: float = 4.0) -> None:
        """Initialize with a characters-per-token ratio.

        Args:
            chars_per_token: Average characters per token.
                Default is 4.0, which works well for English text.
        """
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self._chars_per_token = chars_per_token

    @property
    def chars_per_token(self) -> float:
        """The characters-per-token ratio used for estimation."""
        return self._chars_per_token

    def count_tokens(self, text: str) -> int:
        """Estimate token count from character count.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated number of tokens.
        """
        return int(len(text) / self._chars_per_token)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text based on estimated token limit.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens.

        Returns:
            The truncated text.
        """
        max_chars = int(max_tokens * self._chars_per_token)
        return text[:max_chars]


def get_tokenizer(model: str = "gpt-4") -> Tokenizer:
    """Factory function to get an appropriate tokenizer for a model.

    Attempts to use TiktokenTokenizer for OpenAI models, falling back
    to EstimateTokenizer if tiktoken is not available or the model
    is not recognized.

    Args:
        model: The model name. OpenAI models (containing "gpt" or "text-")
            will use tiktoken if available.

    Returns:
        A Tokenizer instance appropriate for the model.

    Example:
        >>> tokenizer = get_tokenizer("gpt-4")
        >>> isinstance(tokenizer, Tokenizer)
        True
    """
    model_lower = model.lower()

    # Try tiktoken for OpenAI models
    if "gpt" in model_lower or "text-" in model_lower:
        try:
            return TiktokenTokenizer(model)
        except (ImportError, KeyError):
            # tiktoken not installed or model not recognized
            pass

    # Fallback to estimation
    return EstimateTokenizer()
