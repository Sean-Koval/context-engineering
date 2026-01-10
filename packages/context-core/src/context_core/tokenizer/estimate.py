"""Estimation-based tokenizer fallback."""

from __future__ import annotations


class EstimateTokenizer:
    """
    Fallback tokenizer using character-based estimation.

    Provides approximate token counts when exact tokenization
    is not available. Uses a configurable characters-per-token ratio.

    This is useful for:
    - Quick estimates without loading a full tokenizer
    - Models without available tokenizers
    - Testing and development

    Example:
        >>> tokenizer = EstimateTokenizer(chars_per_token=4.0)
        >>> tokenizer.count_tokens("Hello, world!")
        3  # 13 chars / 4.0 = 3.25 -> 3
    """

    # Default ratios for different model families
    DEFAULT_RATIOS: dict[str, float] = {
        "gpt": 4.0,  # OpenAI models
        "claude": 3.5,  # Anthropic Claude models
        "llama": 4.0,  # Meta Llama models
        "mistral": 4.0,  # Mistral models
        "default": 4.0,  # General fallback
    }

    def __init__(
        self,
        chars_per_token: float = 4.0,
        *,
        model_family: str | None = None,
    ) -> None:
        """
        Initialize the estimation tokenizer.

        Args:
            chars_per_token: Average characters per token.
                            Higher values = fewer tokens estimated.
            model_family: Optional model family name to auto-select ratio.
                         If provided, overrides chars_per_token.
        """
        if model_family and model_family.lower() in self.DEFAULT_RATIOS:
            self._chars_per_token = self.DEFAULT_RATIOS[model_family.lower()]
        elif model_family:
            self._chars_per_token = self.DEFAULT_RATIOS["default"]
        else:
            self._chars_per_token = chars_per_token

    @property
    def chars_per_token(self) -> float:
        """The characters-per-token ratio used for estimation."""
        return self._chars_per_token

    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the given text.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated number of tokens (rounded down)
        """
        if not text:
            return 0
        return int(len(text) / self._chars_per_token)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within an estimated token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated text that fits within the estimated token limit.
            Returns original text if already within limit.
        """
        max_chars = int(max_tokens * self._chars_per_token)
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def encode(self, text: str) -> list[int]:
        """
        Create pseudo-token IDs for the text.

        Since this is an estimation tokenizer, it returns
        sequential IDs based on character chunks.

        Args:
            text: The text to encode

        Returns:
            List of pseudo-token IDs (sequential integers)
        """
        if not text:
            return []

        # Create pseudo-tokens based on character chunks
        chunk_size = int(self._chars_per_token)
        num_tokens = (len(text) + chunk_size - 1) // chunk_size
        return list(range(num_tokens))

    def decode(self, tokens: list[int]) -> str:
        """
        Estimate text length from token count.

        Since we don't have actual tokens, this returns
        a placeholder string of estimated length.

        Args:
            tokens: List of token IDs

        Returns:
            Placeholder text of estimated length
        """
        # Return placeholder since we can't actually decode
        estimated_chars = int(len(tokens) * self._chars_per_token)
        return " " * estimated_chars

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Estimate tokens in a list of chat messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Estimated total token count
        """
        # Estimate ~4 tokens overhead per message
        tokens_per_message = 4

        total = 0
        for message in messages:
            total += tokens_per_message
            for value in message.values():
                if isinstance(value, str):
                    total += self.count_tokens(value)

        # Priming tokens
        total += 2

        return total
