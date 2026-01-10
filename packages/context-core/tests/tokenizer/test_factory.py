"""Tests for get_tokenizer factory function."""

from __future__ import annotations

import pytest

from context_core.tokenizer import (
    AnthropicTokenizer,
    EstimateTokenizer,
    Tokenizer,
    get_tokenizer,
)


class TestGetTokenizerFactory:
    """Test get_tokenizer factory function."""

    def test_returns_tokenizer(self) -> None:
        """Factory should return a Tokenizer instance."""
        tokenizer = get_tokenizer("gpt-4")
        assert isinstance(tokenizer, Tokenizer)

    def test_gpt_models(self) -> None:
        """GPT models should return appropriate tokenizer."""
        tokenizer = get_tokenizer("gpt-4")
        # Should be TiktokenTokenizer if tiktoken is installed,
        # otherwise EstimateTokenizer
        assert hasattr(tokenizer, "count_tokens")

    def test_gpt_35_turbo(self) -> None:
        """GPT-3.5-turbo should return appropriate tokenizer."""
        tokenizer = get_tokenizer("gpt-3.5-turbo")
        assert hasattr(tokenizer, "count_tokens")

    def test_claude_models(self) -> None:
        """Claude models should return AnthropicTokenizer."""
        tokenizer = get_tokenizer("claude-3-opus")
        assert isinstance(tokenizer, AnthropicTokenizer)

    def test_claude_3_sonnet(self) -> None:
        """Claude 3 Sonnet should return AnthropicTokenizer."""
        tokenizer = get_tokenizer("claude-3-sonnet")
        assert isinstance(tokenizer, AnthropicTokenizer)

    def test_llama_models(self) -> None:
        """Llama models should return EstimateTokenizer."""
        tokenizer = get_tokenizer("llama-2-70b")
        assert isinstance(tokenizer, EstimateTokenizer)

    def test_mistral_models(self) -> None:
        """Mistral models should return EstimateTokenizer."""
        tokenizer = get_tokenizer("mistral-7b")
        assert isinstance(tokenizer, EstimateTokenizer)

    def test_unknown_model(self) -> None:
        """Unknown model should return EstimateTokenizer."""
        tokenizer = get_tokenizer("totally-unknown-model")
        assert isinstance(tokenizer, EstimateTokenizer)

    def test_case_insensitive(self) -> None:
        """Model names should be case insensitive."""
        tokenizer1 = get_tokenizer("CLAUDE-3-OPUS")
        tokenizer2 = get_tokenizer("claude-3-opus")
        assert type(tokenizer1) is type(tokenizer2)


class TestGetTokenizerWithTiktoken:
    """Tests that require tiktoken to be installed."""

    @pytest.fixture
    def tiktoken_available(self) -> bool:
        """Check if tiktoken is available."""
        from importlib.util import find_spec

        return find_spec("tiktoken") is not None

    def test_gpt_uses_tiktoken_when_available(self, tiktoken_available: bool) -> None:
        """GPT models should use TiktokenTokenizer when available."""
        tokenizer = get_tokenizer("gpt-4")
        if tiktoken_available:
            # Should have tiktoken's encoding_name property
            assert hasattr(tokenizer, "encoding_name")
        else:
            assert isinstance(tokenizer, EstimateTokenizer)


class TestTokenizerFunctionality:
    """Test that tokenizers from factory work correctly."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            "claude-3-opus",
            "llama-2-70b",
            "mistral-7b",
            "unknown-model",
        ],
    )
    def test_count_tokens_works(self, model: str) -> None:
        """All tokenizers should count tokens."""
        tokenizer = get_tokenizer(model)
        count = tokenizer.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            "claude-3-opus",
            "llama-2-70b",
            "mistral-7b",
        ],
    )
    def test_truncate_works(self, model: str) -> None:
        """All tokenizers should truncate text."""
        tokenizer = get_tokenizer(model)
        text = "This is a long text that should be truncated"
        result = tokenizer.truncate_to_tokens(text, max_tokens=3)
        assert isinstance(result, str)
        assert len(result) < len(text)

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            "claude-3-opus",
            "llama-2-70b",
        ],
    )
    def test_encode_works(self, model: str) -> None:
        """All tokenizers should encode text."""
        tokenizer = get_tokenizer(model)
        tokens = tokenizer.encode("Hello, world!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            "claude-3-opus",
            "llama-2-70b",
        ],
    )
    def test_decode_works(self, model: str) -> None:
        """All tokenizers should decode tokens."""
        tokenizer = get_tokenizer(model)
        result = tokenizer.decode([0, 1, 2])
        assert isinstance(result, str)
