"""Tests for Tokenizer protocol."""

from __future__ import annotations

from context_core.tokenizer import EstimateTokenizer, Tokenizer


class TestTokenizerProtocol:
    """Test that tokenizers conform to the Tokenizer protocol."""

    def test_estimate_tokenizer_is_tokenizer(self) -> None:
        """EstimateTokenizer should satisfy Tokenizer protocol."""
        tokenizer = EstimateTokenizer()
        assert isinstance(tokenizer, Tokenizer)

    def test_tokenizer_has_count_tokens(self) -> None:
        """Tokenizer must have count_tokens method."""
        tokenizer = EstimateTokenizer()
        assert hasattr(tokenizer, "count_tokens")
        assert callable(tokenizer.count_tokens)

    def test_tokenizer_has_truncate_to_tokens(self) -> None:
        """Tokenizer must have truncate_to_tokens method."""
        tokenizer = EstimateTokenizer()
        assert hasattr(tokenizer, "truncate_to_tokens")
        assert callable(tokenizer.truncate_to_tokens)

    def test_tokenizer_has_encode(self) -> None:
        """Tokenizer must have encode method."""
        tokenizer = EstimateTokenizer()
        assert hasattr(tokenizer, "encode")
        assert callable(tokenizer.encode)

    def test_tokenizer_has_decode(self) -> None:
        """Tokenizer must have decode method."""
        tokenizer = EstimateTokenizer()
        assert hasattr(tokenizer, "decode")
        assert callable(tokenizer.decode)
