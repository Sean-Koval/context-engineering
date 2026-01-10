"""Tests for AnthropicTokenizer."""

from __future__ import annotations

from context_core.tokenizer import AnthropicTokenizer


class TestAnthropicTokenizerInit:
    """Test AnthropicTokenizer initialization."""

    def test_default_model(self) -> None:
        """Default model should be claude-3-opus."""
        tokenizer = AnthropicTokenizer()
        assert tokenizer.model == "claude-3-opus"

    def test_custom_model(self) -> None:
        """Should accept custom model name."""
        tokenizer = AnthropicTokenizer(model="claude-3-sonnet")
        assert tokenizer.model == "claude-3-sonnet"


class TestAnthropicTokenizerCountTokens:
    """Test count_tokens method."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        tokenizer = AnthropicTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_short_text(self) -> None:
        """Short text should return reasonable estimate."""
        tokenizer = AnthropicTokenizer()
        # Claude uses ~3.5 chars/token
        # "Hello, world!" = 13 chars / 3.5 = ~3.7 -> 3
        result = tokenizer.count_tokens("Hello, world!")
        assert result == 3

    def test_long_text(self) -> None:
        """Long text should return proportional estimate."""
        tokenizer = AnthropicTokenizer()
        text = "a" * 100
        # 100 / 3.5 = ~28.5 -> 28
        result = tokenizer.count_tokens(text)
        assert result == 28


class TestAnthropicTokenizerTruncate:
    """Test truncate_to_tokens method."""

    def test_short_text_unchanged(self) -> None:
        """Text within limit should be unchanged."""
        tokenizer = AnthropicTokenizer()
        text = "Hello"
        result = tokenizer.truncate_to_tokens(text, max_tokens=10)
        assert result == text

    def test_long_text_truncated(self) -> None:
        """Text exceeding limit should be truncated."""
        tokenizer = AnthropicTokenizer()
        text = "a" * 100
        result = tokenizer.truncate_to_tokens(text, max_tokens=10)
        # 10 tokens * 3.5 chars = 35 chars
        assert len(result) == 35


class TestAnthropicTokenizerEncode:
    """Test encode method."""

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        tokenizer = AnthropicTokenizer()
        assert tokenizer.encode("") == []

    def test_returns_list_of_ints(self) -> None:
        """Encode should return list of integers."""
        tokenizer = AnthropicTokenizer()
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)


class TestAnthropicTokenizerDecode:
    """Test decode method."""

    def test_empty_list(self) -> None:
        """Empty token list should return empty string."""
        tokenizer = AnthropicTokenizer()
        assert tokenizer.decode([]) == ""

    def test_returns_string(self) -> None:
        """Decode should return a string."""
        tokenizer = AnthropicTokenizer()
        result = tokenizer.decode([0, 1, 2])
        assert isinstance(result, str)


class TestAnthropicTokenizerCountMessagesTokens:
    """Test count_messages_tokens method."""

    def test_single_message(self) -> None:
        """Single message should include overhead."""
        tokenizer = AnthropicTokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        count = tokenizer.count_messages_tokens(messages)
        # 4 (overhead) + content tokens + 2 (priming)
        assert count >= 6

    def test_multiple_messages(self) -> None:
        """Multiple messages should accumulate overhead."""
        tokenizer = AnthropicTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = tokenizer.count_messages_tokens(messages)
        assert count >= 10

    def test_empty_messages(self) -> None:
        """Empty message list should return priming tokens only."""
        tokenizer = AnthropicTokenizer()
        count = tokenizer.count_messages_tokens([])
        assert count == 2  # Priming tokens only
