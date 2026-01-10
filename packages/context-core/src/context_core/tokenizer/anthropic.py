"""Anthropic tokenizer for Claude models."""

from __future__ import annotations

from .estimate import EstimateTokenizer


class AnthropicTokenizer:
    """
    Tokenizer for Anthropic Claude models.

    Since Anthropic doesn't provide a public tokenizer library,
    this uses estimation with Claude-specific parameters.

    Claude models use a BPE tokenizer with approximately 3.5
    characters per token on average for English text.

    Example:
        >>> tokenizer = AnthropicTokenizer(model="claude-3-opus")
        >>> tokenizer.count_tokens("Hello, world!")
        4
    """

    # Claude model parameters
    # Claude uses ~3.5 chars/token on average
    CHARS_PER_TOKEN = 3.5

    # Token overhead for message formatting
    TOKENS_PER_MESSAGE = 4
    PRIMING_TOKENS = 2

    def __init__(self, model: str = "claude-3-opus") -> None:
        """
        Initialize the Anthropic tokenizer.

        Args:
            model: The Claude model name (for future compatibility)
        """
        self._model = model
        self._estimator = EstimateTokenizer(chars_per_token=self.CHARS_PER_TOKEN)

    @property
    def model(self) -> str:
        """The model name used for tokenization."""
        return self._model

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.

        Args:
            text: The text to tokenize

        Returns:
            Estimated number of tokens
        """
        return self._estimator.count_tokens(text)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated text that fits within the token limit.
        """
        return self._estimator.truncate_to_tokens(text, max_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Create pseudo-token IDs for the text.

        Since we don't have access to Claude's actual tokenizer,
        this returns sequential IDs based on estimated chunks.

        Args:
            text: The text to encode

        Returns:
            List of pseudo-token IDs
        """
        return self._estimator.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        Estimate text from token count.

        Args:
            tokens: List of token IDs

        Returns:
            Placeholder text of estimated length
        """
        return self._estimator.decode(tokens)

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Estimate tokens in a list of chat messages.

        Accounts for Claude's message formatting overhead.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Estimated total token count
        """
        total = 0

        for message in messages:
            total += self.TOKENS_PER_MESSAGE
            for value in message.values():
                if isinstance(value, str):
                    total += self.count_tokens(value)

        # Priming tokens
        total += self.PRIMING_TOKENS

        return total
