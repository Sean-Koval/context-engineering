"""Tokenizer module - Model-agnostic token counting.

Components:
- Tokenizer: Protocol for tokenizer implementations
- TiktokenTokenizer: OpenAI tiktoken implementation
- AnthropicTokenizer: Anthropic Claude estimation
- EstimateTokenizer: Fallback estimation

Example:
    >>> from context_core.tokenizer import get_tokenizer
    >>> tokenizer = get_tokenizer("gpt-4")
    >>> tokenizer.count_tokens("Hello, world!")
    4
"""

from __future__ import annotations

from .anthropic import AnthropicTokenizer
from .base import Tokenizer
from .estimate import EstimateTokenizer

__all__ = [
    "Tokenizer",
    "TiktokenTokenizer",
    "AnthropicTokenizer",
    "EstimateTokenizer",
    "get_tokenizer",
]


def _get_tiktoken_tokenizer() -> type | None:
    """Lazy import of TiktokenTokenizer to avoid import errors."""
    try:
        from .tiktoken import TiktokenTokenizer

        return TiktokenTokenizer
    except ImportError:
        return None


# Make TiktokenTokenizer available at module level
TiktokenTokenizer: type | None = None


def __getattr__(name: str) -> type:
    """Lazy loading for TiktokenTokenizer."""
    global TiktokenTokenizer  # noqa: PLW0603
    if name == "TiktokenTokenizer":
        if TiktokenTokenizer is None:
            TiktokenTokenizer = _get_tiktoken_tokenizer()
        if TiktokenTokenizer is None:
            msg = (
                "TiktokenTokenizer requires tiktoken. "
                "Install with: pip install context-core[tiktoken]"
            )
            raise ImportError(msg)
        return TiktokenTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_tokenizer(model: str) -> Tokenizer:
    """
    Factory function to get appropriate tokenizer for a model.

    Attempts to use the most accurate tokenizer available for the model.
    Falls back to estimation if specific tokenizers are unavailable.

    Args:
        model: Model name (e.g., "gpt-4", "claude-3-opus", "llama-2")

    Returns:
        A Tokenizer instance appropriate for the model

    Example:
        >>> tokenizer = get_tokenizer("gpt-4")
        >>> tokenizer.count_tokens("Hello")
        1

        >>> tokenizer = get_tokenizer("claude-3-opus")
        >>> tokenizer.count_tokens("Hello")
        2
    """
    model_lower = model.lower()

    # OpenAI models - use tiktoken if available
    if any(name in model_lower for name in ("gpt", "text-", "davinci", "turbo")):
        tiktoken_cls = _get_tiktoken_tokenizer()
        if tiktoken_cls is not None:
            try:
                return tiktoken_cls(model)
            except (ImportError, KeyError):
                pass
        # Fall back to estimation for OpenAI
        return EstimateTokenizer(model_family="gpt")

    # Anthropic models
    if any(name in model_lower for name in ("claude", "anthropic")):
        return AnthropicTokenizer(model)

    # Llama models
    if any(name in model_lower for name in ("llama", "meta")):
        return EstimateTokenizer(model_family="llama")

    # Mistral models
    if "mistral" in model_lower:
        return EstimateTokenizer(model_family="mistral")

    # Default fallback
    return EstimateTokenizer()
