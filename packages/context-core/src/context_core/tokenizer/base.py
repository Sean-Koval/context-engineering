"""Base tokenizer protocol for model-agnostic token counting."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """
    Protocol for tokenizer implementations.

    Tokenizers provide model-agnostic token counting and truncation.
    Implementations should be thread-safe and may use caching for performance.

    Example:
        >>> tokenizer = TiktokenTokenizer(model="gpt-4")
        >>> count = tokenizer.count_tokens("Hello, world!")
        >>> truncated = tokenizer.truncate_to_tokens("Long text...", max_tokens=10)
    """

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens in the text
        """
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated text that fits within the token limit
        """
        ...

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        ...

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        ...
