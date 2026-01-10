"""Tests for EstimateTokenizer."""

from __future__ import annotations

from context_core.tokenizer import EstimateTokenizer


class TestEstimateTokenizerInit:
    """Test EstimateTokenizer initialization."""

    def test_default_chars_per_token(self) -> None:
        """Default chars_per_token should be 4.0."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.chars_per_token == 4.0

    def test_custom_chars_per_token(self) -> None:
        """Should accept custom chars_per_token."""
        tokenizer = EstimateTokenizer(chars_per_token=3.0)
        assert tokenizer.chars_per_token == 3.0

    def test_model_family_gpt(self) -> None:
        """GPT model family should use 4.0 ratio."""
        tokenizer = EstimateTokenizer(model_family="gpt")
        assert tokenizer.chars_per_token == 4.0

    def test_model_family_claude(self) -> None:
        """Claude model family should use 3.5 ratio."""
        tokenizer = EstimateTokenizer(model_family="claude")
        assert tokenizer.chars_per_token == 3.5

    def test_model_family_unknown(self) -> None:
        """Unknown model family should use default ratio."""
        tokenizer = EstimateTokenizer(model_family="unknown-model")
        assert tokenizer.chars_per_token == 4.0


class TestEstimateTokenizerCountTokens:
    """Test count_tokens method."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_short_text(self) -> None:
        """Short text should return correct estimate."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        # 12 chars / 4.0 = 3 tokens
        assert tokenizer.count_tokens("Hello, world") == 3

    def test_long_text(self) -> None:
        """Long text should return correct estimate."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "a" * 100
        # 100 chars / 4.0 = 25 tokens
        assert tokenizer.count_tokens(text) == 25

    def test_rounds_down(self) -> None:
        """Token count should round down."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        # 13 chars / 4.0 = 3.25 -> 3
        assert tokenizer.count_tokens("Hello, world!") == 3


class TestEstimateTokenizerTruncate:
    """Test truncate_to_tokens method."""

    def test_short_text_unchanged(self) -> None:
        """Text within limit should be unchanged."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "Hello"
        result = tokenizer.truncate_to_tokens(text, max_tokens=10)
        assert result == text

    def test_long_text_truncated(self) -> None:
        """Text exceeding limit should be truncated."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "a" * 100
        result = tokenizer.truncate_to_tokens(text, max_tokens=10)
        # 10 tokens * 4.0 chars/token = 40 chars
        assert len(result) == 40

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.truncate_to_tokens("", max_tokens=10) == ""

    def test_zero_tokens(self) -> None:
        """Zero max_tokens should return empty string."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.truncate_to_tokens("Hello, world!", max_tokens=0) == ""


class TestEstimateTokenizerEncode:
    """Test encode method."""

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.encode("") == []

    def test_returns_sequential_ids(self) -> None:
        """Encode should return sequential IDs."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "Hello, world!"  # 13 chars -> ~4 tokens
        tokens = tokenizer.encode(text)
        # Should be sequential starting from 0
        assert tokens == list(range(len(tokens)))

    def test_encode_length_matches_count(self) -> None:
        """Encode length should approximately match count_tokens."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        count = tokenizer.count_tokens(text)
        # They should be close (might differ by 1 due to rounding)
        assert abs(len(tokens) - count) <= 1


class TestEstimateTokenizerDecode:
    """Test decode method."""

    def test_empty_list(self) -> None:
        """Empty token list should return empty string."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.decode([]) == ""

    def test_returns_placeholder(self) -> None:
        """Decode should return placeholder of estimated length."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        tokens = [0, 1, 2, 3, 4]  # 5 tokens
        result = tokenizer.decode(tokens)
        # 5 tokens * 4.0 chars = 20 chars
        assert len(result) == 20


class TestEstimateTokenizerCountMessagesTokens:
    """Test count_messages_tokens method."""

    def test_single_message(self) -> None:
        """Single message should include overhead."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        messages = [{"role": "user", "content": "Hello"}]
        count = tokenizer.count_messages_tokens(messages)
        # 4 (message overhead) + 1 (content ~1 token) + 2 (priming) + role ~1 token
        assert count >= 6

    def test_multiple_messages(self) -> None:
        """Multiple messages should accumulate overhead."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = tokenizer.count_messages_tokens(messages)
        # Each message has overhead
        assert count >= 10

    def test_empty_messages(self) -> None:
        """Empty message list should return priming tokens only."""
        tokenizer = EstimateTokenizer()
        count = tokenizer.count_messages_tokens([])
        assert count == 2  # Just priming tokens
