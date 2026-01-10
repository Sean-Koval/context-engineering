"""Tests for TiktokenTokenizer."""

from __future__ import annotations

import pytest

# Skip all tests if tiktoken is not installed
tiktoken = pytest.importorskip("tiktoken")

from context_core.tokenizer import TiktokenTokenizer  # noqa: E402


class TestTiktokenTokenizerInit:
    """Test TiktokenTokenizer initialization."""

    def test_default_model(self) -> None:
        """Default model should be gpt-4."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.model == "gpt-4"

    def test_custom_model(self) -> None:
        """Should accept custom model name."""
        tokenizer = TiktokenTokenizer(model="gpt-3.5-turbo")
        assert tokenizer.model == "gpt-3.5-turbo"

    def test_unknown_model_falls_back(self) -> None:
        """Unknown model should fall back to cl100k_base."""
        tokenizer = TiktokenTokenizer(model="unknown-model-xyz")
        assert tokenizer.encoding_name == "cl100k_base"


class TestTiktokenTokenizerCountTokens:
    """Test count_tokens method."""

    def test_empty_string(self) -> None:
        """Empty string should return 0 tokens."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_hello_world(self) -> None:
        """'Hello, world!' should be 4 tokens with GPT-4."""
        tokenizer = TiktokenTokenizer(model="gpt-4")
        assert tokenizer.count_tokens("Hello, world!") == 4

    def test_repeated_calls_consistent(self) -> None:
        """Repeated calls should return consistent results."""
        tokenizer = TiktokenTokenizer()
        text = "Hello, world!"
        count1 = tokenizer.count_tokens(text)
        count2 = tokenizer.count_tokens(text)
        assert count1 == count2


class TestTiktokenTokenizerTruncate:
    """Test truncate_to_tokens method."""

    def test_short_text_unchanged(self) -> None:
        """Text within limit should be unchanged."""
        tokenizer = TiktokenTokenizer()
        text = "Hello"
        result = tokenizer.truncate_to_tokens(text, max_tokens=10)
        assert result == text

    def test_long_text_truncated(self) -> None:
        """Text exceeding limit should be truncated."""
        tokenizer = TiktokenTokenizer()
        text = "This is a longer text that will be truncated to fewer tokens"
        result = tokenizer.truncate_to_tokens(text, max_tokens=5)
        # Result should have exactly 5 tokens
        assert tokenizer.count_tokens(result) == 5

    def test_truncation_preserves_start(self) -> None:
        """Truncation should preserve the start of the text."""
        tokenizer = TiktokenTokenizer()
        text = "Hello world, how are you doing today?"
        result = tokenizer.truncate_to_tokens(text, max_tokens=3)
        assert result.startswith("Hello")


class TestTiktokenTokenizerEncode:
    """Test encode method."""

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.encode("") == []

    def test_returns_integers(self) -> None:
        """Encode should return list of integers."""
        tokenizer = TiktokenTokenizer()
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_length_matches_count(self) -> None:
        """Encode length should match count_tokens."""
        tokenizer = TiktokenTokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        count = tokenizer.count_tokens(text)
        assert len(tokens) == count


class TestTiktokenTokenizerDecode:
    """Test decode method."""

    def test_empty_list(self) -> None:
        """Empty token list should return empty string."""
        tokenizer = TiktokenTokenizer()
        assert tokenizer.decode([]) == ""

    def test_roundtrip(self) -> None:
        """Encode then decode should return original text."""
        tokenizer = TiktokenTokenizer()
        original = "Hello, world!"
        tokens = tokenizer.encode(original)
        decoded = tokenizer.decode(tokens)
        assert decoded == original


class TestTiktokenTokenizerCountMessagesTokens:
    """Test count_messages_tokens method."""

    def test_single_message(self) -> None:
        """Single message should include overhead."""
        tokenizer = TiktokenTokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        count = tokenizer.count_messages_tokens(messages)
        # Should be: content token + "user" role + overhead
        assert count >= 4

    def test_multiple_messages(self) -> None:
        """Multiple messages should accumulate overhead."""
        tokenizer = TiktokenTokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = tokenizer.count_messages_tokens(messages)
        # Each message adds overhead
        assert count >= 8

    def test_empty_messages(self) -> None:
        """Empty message list should return priming tokens only."""
        tokenizer = TiktokenTokenizer()
        count = tokenizer.count_messages_tokens([])
        assert count == 2  # Just priming tokens
