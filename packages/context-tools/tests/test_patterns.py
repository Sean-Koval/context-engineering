"""Tests for ToolUsagePatterns pattern detection."""

from __future__ import annotations

import pytest

from context_tools import ToolCallSignature, ToolPattern
from context_tools.patterns import (
    ParameterStats,
    ToolSequence,
    ToolUsagePatterns,
    UsageStats,
)


class TestToolUsagePatterns:
    """Tests for ToolUsagePatterns class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        patterns = ToolUsagePatterns()
        assert patterns.history_size == 0
        assert patterns.transition_count == 0

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        patterns = ToolUsagePatterns(
            window_size=100,
            min_pattern_frequency=5,
            max_sequence_length=3,
        )
        assert patterns._window_size == 100
        assert patterns._min_frequency == 5
        assert patterns._max_sequence_length == 3

    def test_record_single_call(self) -> None:
        """Test recording a single tool call."""
        patterns = ToolUsagePatterns()
        sig = ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})

        patterns.record(sig)

        assert patterns.history_size == 1
        assert patterns.transition_count == 0  # No transition yet

    def test_record_creates_transitions(self) -> None:
        """Test that recording builds transition matrix."""
        patterns = ToolUsagePatterns()

        patterns.record(ToolCallSignature(tool_name="search", arguments={}))
        patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        assert patterns.history_size == 2
        assert patterns.transition_count == 1
        assert patterns._transitions["search"]["read_file"] == 1

    def test_record_window_trimming(self) -> None:
        """Test that history is trimmed to window size."""
        patterns = ToolUsagePatterns(window_size=5)

        for i in range(10):
            patterns.record(ToolCallSignature(tool_name=f"tool_{i}", arguments={}))

        assert patterns.history_size == 5

    def test_predict_next_tool_empty_history(self) -> None:
        """Test prediction with no history returns empty."""
        patterns = ToolUsagePatterns()
        predictions = patterns.predict_next_tool("unknown_tool")
        assert predictions == []

    def test_predict_next_tool_single_transition(self) -> None:
        """Test prediction with single transition."""
        patterns = ToolUsagePatterns()

        patterns.record(ToolCallSignature(tool_name="search", arguments={}))
        patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        predictions = patterns.predict_next_tool("search")

        assert len(predictions) == 1
        assert predictions[0][0] == "read_file"
        assert predictions[0][1] == 1.0

    def test_predict_next_tool_multiple_transitions(self) -> None:
        """Test prediction with multiple possible transitions."""
        patterns = ToolUsagePatterns()

        # search -> read_file (3 times)
        for _ in range(3):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        # search -> write_file (1 time)
        patterns.record(ToolCallSignature(tool_name="search", arguments={}))
        patterns.record(ToolCallSignature(tool_name="write_file", arguments={}))

        predictions = patterns.predict_next_tool("search", top_k=2)

        assert len(predictions) == 2
        assert predictions[0][0] == "read_file"
        assert predictions[0][1] == 0.75  # 3/4
        assert predictions[1][0] == "write_file"
        assert predictions[1][1] == 0.25  # 1/4

    def test_predict_next_tool_with_context_boost(self) -> None:
        """Test that recent context boosts predictions."""
        patterns = ToolUsagePatterns()

        # Equal transitions to read_file and edit
        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="edit", arguments={}))

        # With edit in recent context, it should be boosted
        predictions = patterns.predict_next_tool(
            "search", context_tools=["edit"], top_k=2
        )

        # edit should be boosted above read_file
        assert predictions[0][0] == "edit"

    def test_detect_sequences_min_frequency(self) -> None:
        """Test sequence detection respects min frequency."""
        patterns = ToolUsagePatterns(min_pattern_frequency=3)

        # Create a sequence that appears 5 times
        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        sequences = patterns.detect_sequences(max_length=2)

        assert len(sequences) >= 1
        seq = next((s for s in sequences if s.tools == ["search", "read_file"]), None)
        assert seq is not None
        assert seq.frequency == 5

    def test_detect_sequences_respects_max_length(self) -> None:
        """Test sequence detection respects max length."""
        patterns = ToolUsagePatterns(min_pattern_frequency=1)

        for _ in range(3):
            patterns.record(ToolCallSignature(tool_name="a", arguments={}))
            patterns.record(ToolCallSignature(tool_name="b", arguments={}))
            patterns.record(ToolCallSignature(tool_name="c", arguments={}))
            patterns.record(ToolCallSignature(tool_name="d", arguments={}))

        # With max_length=2, should only get 2-tool sequences
        sequences = patterns.detect_sequences(max_length=2)
        for seq in sequences:
            assert len(seq.tools) == 2

        # With max_length=3, should get longer sequences
        sequences = patterns.detect_sequences(max_length=3)
        has_length_3 = any(len(s.tools) == 3 for s in sequences)
        assert has_length_3

    def test_parameter_pattern_learning(self) -> None:
        """Test parameter value tracking."""
        patterns = ToolUsagePatterns()

        # Record calls with same parameter values
        for _ in range(5):
            patterns.record(
                ToolCallSignature(
                    tool_name="read_file", arguments={"path": "/common/path.py"}
                )
            )
        for _ in range(2):
            patterns.record(
                ToolCallSignature(
                    tool_name="read_file", arguments={"path": "/other/file.py"}
                )
            )

        stats = patterns.get_parameter_pattern("read_file", "path")
        assert stats is not None
        assert stats.total_observations == 7
        assert stats.most_common_value() == "/common/path.py"

    def test_predict_arguments(self) -> None:
        """Test argument prediction from patterns."""
        patterns = ToolUsagePatterns()

        # Build up pattern data
        for _ in range(10):
            patterns.record(
                ToolCallSignature(
                    tool_name="read_file",
                    arguments={"path": "/src/main.py", "encoding": "utf-8"},
                )
            )

        predicted = patterns.predict_arguments("read_file")
        assert predicted["path"] == "/src/main.py"
        assert predicted["encoding"] == "utf-8"

    def test_predict_arguments_preserves_partial(self) -> None:
        """Test that partial args aren't overwritten."""
        patterns = ToolUsagePatterns()

        for _ in range(10):
            patterns.record(
                ToolCallSignature(
                    tool_name="read_file", arguments={"path": "/common.py"}
                )
            )

        predicted = patterns.predict_arguments(
            "read_file", partial_args={"path": "/override.py"}
        )
        assert predicted.get("path") is None  # Don't override existing

    def test_get_transition_probability(self) -> None:
        """Test getting transition probability."""
        patterns = ToolUsagePatterns()

        for _ in range(3):
            patterns.record(ToolCallSignature(tool_name="a", arguments={}))
            patterns.record(ToolCallSignature(tool_name="b", arguments={}))
        patterns.record(ToolCallSignature(tool_name="a", arguments={}))
        patterns.record(ToolCallSignature(tool_name="c", arguments={}))

        prob_ab = patterns.get_transition_probability("a", "b")
        prob_ac = patterns.get_transition_probability("a", "c")
        prob_unknown = patterns.get_transition_probability("x", "y")

        assert prob_ab == 0.75  # 3/4
        assert prob_ac == 0.25  # 1/4
        assert prob_unknown == 0.0

    def test_get_tool_frequency(self) -> None:
        """Test tool frequency counting."""
        patterns = ToolUsagePatterns()

        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
        for _ in range(3):
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        assert patterns.get_tool_frequency("search") == 5
        assert patterns.get_tool_frequency("read_file") == 3
        assert patterns.get_tool_frequency("unknown") == 0

    def test_get_stats(self) -> None:
        """Test usage statistics generation."""
        patterns = ToolUsagePatterns(min_pattern_frequency=1)

        for _ in range(3):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        stats = patterns.get_stats()

        assert isinstance(stats, UsageStats)
        assert stats.history_size == 6
        assert stats.unique_tools == 2
        assert stats.tool_frequencies["search"] == 3
        assert stats.tool_frequencies["read_file"] == 3
        assert stats.transition_count > 0

    def test_clear(self) -> None:
        """Test clearing all data."""
        patterns = ToolUsagePatterns()

        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="test", arguments={"a": 1}))

        assert patterns.history_size > 0

        patterns.clear()

        assert patterns.history_size == 0
        assert patterns.transition_count == 0
        assert len(patterns._parameter_patterns) == 0

    def test_to_patterns_conversion(self) -> None:
        """Test conversion to ToolPattern objects."""
        patterns = ToolUsagePatterns(min_pattern_frequency=2)

        for _ in range(5):
            patterns.record(ToolCallSignature(tool_name="search", arguments={}))
            patterns.record(ToolCallSignature(tool_name="read_file", arguments={}))

        tool_patterns = patterns.to_patterns()

        assert len(tool_patterns) > 0
        assert all(isinstance(p, ToolPattern) for p in tool_patterns)

        # Find our sequence
        seq_pattern = next(
            (p for p in tool_patterns if p.sequence == ["search", "read_file"]), None
        )
        assert seq_pattern is not None
        assert seq_pattern.frequency == 5
        assert seq_pattern.confidence == 0.5  # 5/10


class TestToolSequence:
    """Tests for ToolSequence model."""

    def test_basic_creation(self) -> None:
        """Test basic sequence creation."""
        seq = ToolSequence(tools=["a", "b"], frequency=3, avg_gap_ms=100.0)
        assert seq.tools == ["a", "b"]
        assert seq.frequency == 3
        assert seq.avg_gap_ms == 100.0

    def test_min_length_validation(self) -> None:
        """Test that sequences require at least 2 tools."""
        with pytest.raises(ValueError):
            ToolSequence(tools=["a"])  # Too short


class TestParameterStats:
    """Tests for ParameterStats model."""

    def test_record_value(self) -> None:
        """Test recording values."""
        stats = ParameterStats(param_name="path")
        stats.record_value("/a.py")
        stats.record_value("/a.py")
        stats.record_value("/b.py")

        assert stats.total_observations == 3
        assert stats.value_counts["/a.py"] == 2
        assert stats.value_counts["/b.py"] == 1

    def test_most_common_value(self) -> None:
        """Test getting most common value."""
        stats = ParameterStats(param_name="path")
        stats.record_value("/common.py")
        stats.record_value("/common.py")
        stats.record_value("/common.py")
        stats.record_value("/rare.py")

        assert stats.most_common_value() == "/common.py"

    def test_most_common_empty(self) -> None:
        """Test most common returns None when empty."""
        stats = ParameterStats(param_name="path")
        assert stats.most_common_value() is None

    def test_ignores_long_values(self) -> None:
        """Test that very long values aren't tracked."""
        stats = ParameterStats(param_name="content")
        long_value = "x" * 200  # Too long
        stats.record_value(long_value)

        assert stats.total_observations == 1
        assert len(stats.value_counts) == 0  # Not tracked


class TestUsageStats:
    """Tests for UsageStats model."""

    def test_defaults(self) -> None:
        """Test default values."""
        stats = UsageStats()
        assert stats.history_size == 0
        assert stats.unique_tools == 0
        assert stats.tool_frequencies == {}
