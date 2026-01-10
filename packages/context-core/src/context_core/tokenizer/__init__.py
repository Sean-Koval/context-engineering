"""Tokenizer module - Model-agnostic token counting.

Components:
- Tokenizer: Protocol for tokenizer implementations
- TiktokenTokenizer: OpenAI tiktoken implementation
- EstimateTokenizer: Fallback estimation
"""

from __future__ import annotations
