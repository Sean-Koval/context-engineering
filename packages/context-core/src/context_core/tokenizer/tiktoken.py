"""Tiktoken tokenizer for OpenAI models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tiktoken


class TiktokenTokenizer:
    """
    Tokenizer using tiktoken for OpenAI models.

    Provides accurate token counting for GPT-3.5, GPT-4, and other
    OpenAI models. Requires the 'tiktoken' optional dependency.

    Example:
        >>> tokenizer = TiktokenTokenizer(model="gpt-4")
        >>> tokenizer.count_tokens("Hello, world!")
        4
        >>> tokenizer.truncate_to_tokens("This is a long text", max_tokens=3)
        'This is a'
    """

    def __init__(self, model: str = "gpt-4") -> None:
        """
        Initialize the tiktoken tokenizer.

        Args:
            model: The model name to use for tokenization.
                   Supported: gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.

        Raises:
            ImportError: If tiktoken is not installed
            KeyError: If the model is not recognized
        """
        try:
            import tiktoken as _tiktoken
        except ImportError as e:
            msg = (
                "tiktoken is required for TiktokenTokenizer. "
                "Install with: pip install context-core[tiktoken]"
            )
            raise ImportError(msg) from e

        self._model = model
        try:
            self._encoding: tiktoken.Encoding = _tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            self._encoding = _tiktoken.get_encoding("cl100k_base")

    @property
    def model(self) -> str:
        """The model name used for tokenization."""
        return self._model

    @property
    def encoding_name(self) -> str:
        """The encoding name used by this tokenizer."""
        return self._encoding.name

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to tokenize

        Returns:
            Number of tokens in the text
        """
        return len(self._encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token limit.

        Args:
            text: The text to truncate
            max_tokens: Maximum number of tokens allowed

        Returns:
            Truncated text that fits within the token limit.
            Returns original text if already within limit.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.

        Args:
            text: The text to encode

        Returns:
            List of token IDs
        """
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """
        Decode token IDs back into text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        return self._encoding.decode(tokens)

    def count_messages_tokens(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.

        Accounts for special tokens used in chat format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Total token count including special tokens
        """
        # Token overhead per message varies by model
        # GPT-4/GPT-3.5-turbo use ~4 tokens per message
        tokens_per_message = 4

        total = 0
        for message in messages:
            total += tokens_per_message
            for _key, value in message.items():
                if isinstance(value, str):
                    total += self.count_tokens(value)

        # Every reply is primed with assistant
        total += 2

        return total
