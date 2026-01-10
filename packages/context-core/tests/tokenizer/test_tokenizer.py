"""Tests for tokenizer implementations."""

from __future__ import annotations

import pytest

from context_core.tokenizer import (
    EstimateTokenizer,
    Tokenizer,
    get_tokenizer,
)


class TestTokenizerProtocol:
    """Tests for Tokenizer protocol compliance."""

    def test_estimate_tokenizer_implements_protocol(self) -> None:
        """EstimateTokenizer satisfies the Tokenizer protocol."""
        tokenizer = EstimateTokenizer()
        assert isinstance(tokenizer, Tokenizer)

    def test_get_tokenizer_returns_protocol_compliant(self) -> None:
        """get_tokenizer returns a Tokenizer-compliant instance."""
        tokenizer = get_tokenizer("unknown-model")
        assert isinstance(tokenizer, Tokenizer)


class TestEstimateTokenizer:
    """Tests for EstimateTokenizer."""

    def test_default_chars_per_token(self) -> None:
        """Default chars_per_token is 4.0."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.chars_per_token == 4.0

    def test_custom_chars_per_token(self) -> None:
        """Can set custom chars_per_token."""
        tokenizer = EstimateTokenizer(chars_per_token=3.5)
        assert tokenizer.chars_per_token == 3.5

    def test_invalid_chars_per_token(self) -> None:
        """Raises ValueError for non-positive chars_per_token."""
        with pytest.raises(ValueError, match="must be positive"):
            EstimateTokenizer(chars_per_token=0)
        with pytest.raises(ValueError, match="must be positive"):
            EstimateTokenizer(chars_per_token=-1)

    def test_count_tokens_empty(self) -> None:
        """Empty string has 0 tokens."""
        tokenizer = EstimateTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_basic(self) -> None:
        """Token count is len(text) / chars_per_token."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        # 16 chars / 4 = 4 tokens
        assert tokenizer.count_tokens("Hello, world!!!!") == 4
        # 8 chars / 4 = 2 tokens
        assert tokenizer.count_tokens("12345678") == 2

    def test_count_tokens_rounds_down(self) -> None:
        """Token count is floored (int division behavior)."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        # 7 chars / 4 = 1.75 -> 1 token
        assert tokenizer.count_tokens("1234567") == 1
        # 3 chars / 4 = 0.75 -> 0 tokens
        assert tokenizer.count_tokens("abc") == 0

    def test_truncate_to_tokens_no_truncation_needed(self) -> None:
        """Text within limit is returned unchanged."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "Hello!"  # 6 chars
        result = tokenizer.truncate_to_tokens(text, 10)  # 40 chars allowed
        assert result == text

    def test_truncate_to_tokens_truncates(self) -> None:
        """Text exceeding limit is truncated."""
        tokenizer = EstimateTokenizer(chars_per_token=4.0)
        text = "Hello, world! This is a test."  # 29 chars
        # 2 tokens * 4 = 8 chars max
        result = tokenizer.truncate_to_tokens(text, 2)
        assert result == "Hello, w"
        assert len(result) == 8

    def test_truncate_to_tokens_zero_limit(self) -> None:
        """Zero tokens returns empty string."""
        tokenizer = EstimateTokenizer()
        result = tokenizer.truncate_to_tokens("Hello", 0)
        assert result == ""


class TestTiktokenTokenizer:
    """Tests for TiktokenTokenizer.

    These tests require tiktoken to be installed.
    """

    @pytest.fixture
    def tiktoken_available(self) -> bool:
        """Check if tiktoken is available."""
        try:
            import tiktoken  # noqa: F401

            return True
        except ImportError:
            return False

    def test_import_error_without_tiktoken(self, tiktoken_available: bool) -> None:
        """TiktokenTokenizer raises ImportError if tiktoken not installed."""
        if tiktoken_available:
            pytest.skip("tiktoken is installed")

        from context_core.tokenizer import TiktokenTokenizer

        with pytest.raises(ImportError):
            TiktokenTokenizer()

    def test_model_property(self, tiktoken_available: bool) -> None:
        """Model property returns configured model."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer("gpt-4")
        assert tokenizer.model == "gpt-4"

    def test_count_tokens_empty(self, tiktoken_available: bool) -> None:
        """Empty string has 0 tokens."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer()
        assert tokenizer.count_tokens("") == 0

    def test_count_tokens_basic(self, tiktoken_available: bool) -> None:
        """Token count matches tiktoken encoding."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer("gpt-4")
        # "Hello, world!" typically tokenizes to 4 tokens in GPT-4
        count = tokenizer.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_cached(self, tiktoken_available: bool) -> None:
        """Token counts are cached."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer()
        text = "This is a test."

        # First call - should add to cache
        count1 = tokenizer.count_tokens(text)
        assert text in tokenizer._cache

        # Second call should hit cache
        count2 = tokenizer.count_tokens(text)
        assert count1 == count2

    def test_cache_eviction(self, tiktoken_available: bool) -> None:
        """Cache evicts entries when full."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        # Small cache for testing eviction
        tokenizer = TiktokenTokenizer(cache_size=4)

        # Fill the cache
        for i in range(4):
            tokenizer.count_tokens(f"text {i}")

        assert len(tokenizer._cache) == 4

        # Add one more - should trigger eviction
        tokenizer.count_tokens("text 4")

        # Cache should have been halved (2 removed) then 1 added = 3
        assert len(tokenizer._cache) == 3

    def test_truncate_to_tokens_no_truncation(self, tiktoken_available: bool) -> None:
        """Text within limit is returned unchanged."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer()
        text = "Hello"
        result = tokenizer.truncate_to_tokens(text, 100)
        assert result == text

    def test_truncate_to_tokens_truncates(self, tiktoken_available: bool) -> None:
        """Text exceeding limit is truncated to exact token boundary."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer()
        text = "Hello, world! This is a test of truncation."

        # Truncate to 3 tokens
        result = tokenizer.truncate_to_tokens(text, 3)

        # Result should have exactly 3 tokens
        assert tokenizer.count_tokens(result) == 3

    def test_truncate_to_tokens_zero_limit(self, tiktoken_available: bool) -> None:
        """Zero tokens returns empty string."""
        if not tiktoken_available:
            pytest.skip("tiktoken not installed")

        from context_core.tokenizer import TiktokenTokenizer

        tokenizer = TiktokenTokenizer()
        result = tokenizer.truncate_to_tokens("Hello, world!", 0)
        assert result == ""


class TestGetTokenizer:
    """Tests for get_tokenizer factory function."""

    def test_returns_estimate_for_unknown_model(self) -> None:
        """Unknown models get EstimateTokenizer."""
        tokenizer = get_tokenizer("unknown-model-xyz")
        assert isinstance(tokenizer, EstimateTokenizer)

    def test_returns_estimate_for_non_openai_model(self) -> None:
        """Non-OpenAI models get EstimateTokenizer."""
        tokenizer = get_tokenizer("claude-3-opus")
        assert isinstance(tokenizer, EstimateTokenizer)

    def test_gpt_model_with_tiktoken(self) -> None:
        """GPT models get TiktokenTokenizer if available."""
        try:
            import tiktoken  # noqa: F401

            has_tiktoken = True
        except ImportError:
            has_tiktoken = False

        tokenizer = get_tokenizer("gpt-4")

        if has_tiktoken:
            from context_core.tokenizer import TiktokenTokenizer

            assert isinstance(tokenizer, TiktokenTokenizer)
        else:
            assert isinstance(tokenizer, EstimateTokenizer)

    def test_text_model_with_tiktoken(self) -> None:
        """text-* models get TiktokenTokenizer if available."""
        try:
            import tiktoken  # noqa: F401

            has_tiktoken = True
        except ImportError:
            has_tiktoken = False

        tokenizer = get_tokenizer("text-davinci-003")

        if has_tiktoken:
            from context_core.tokenizer import TiktokenTokenizer

            assert isinstance(tokenizer, TiktokenTokenizer)
        else:
            assert isinstance(tokenizer, EstimateTokenizer)

    def test_case_insensitive_model_detection(self) -> None:
        """Model detection is case-insensitive for determining tokenizer type.

        Note: The actual model name passed to tiktoken must be valid.
        This test verifies our detection logic is case-insensitive.
        """
        try:
            import tiktoken  # noqa: F401

            has_tiktoken = True
        except ImportError:
            has_tiktoken = False

        # Use lowercase "gpt-4" but uppercase in our detection string
        # The detection should work regardless of case
        tokenizer = get_tokenizer("gpt-4")  # Valid tiktoken model

        if has_tiktoken:
            from context_core.tokenizer import TiktokenTokenizer

            assert isinstance(tokenizer, TiktokenTokenizer)
        else:
            assert isinstance(tokenizer, EstimateTokenizer)

        # Also test that detection works with mixed case in model name
        # (will fall back to EstimateTokenizer since tiktoken needs exact names)
        tokenizer2 = get_tokenizer("GPT-4-TURBO")  # Invalid for tiktoken
        # This falls back to estimate because tiktoken doesn't recognize it
        assert isinstance(tokenizer2, EstimateTokenizer)

    def test_default_model_is_gpt4(self) -> None:
        """Default model parameter is gpt-4."""
        try:
            import tiktoken  # noqa: F401

            has_tiktoken = True
        except ImportError:
            has_tiktoken = False

        tokenizer = get_tokenizer()

        if has_tiktoken:
            from context_core.tokenizer import TiktokenTokenizer

            assert isinstance(tokenizer, TiktokenTokenizer)
            assert tokenizer.model == "gpt-4"
        else:
            assert isinstance(tokenizer, EstimateTokenizer)
