"""Tests for list truncation with smart sampling.

Tests cover:
- TruncationStrategy enum
- StatisticalSummary, TypeDistribution, TruncationResult models
- ListTruncator: All sampling strategies
- Integration with ToolResultCompressor
"""

from __future__ import annotations

import pytest

from context_tools import (
    ListTruncator,
    StatisticalSummary,
    ToolResultCompressor,
    TruncationResult,
    TruncationStrategy,
    TypeDistribution,
)


class TestTruncationStrategy:
    """Tests for TruncationStrategy enum."""

    def test_enum_values(self) -> None:
        """Test all enum values exist."""
        assert TruncationStrategy.HEAD_TAIL == "head_tail"
        assert TruncationStrategy.UNIFORM == "uniform"
        assert TruncationStrategy.RESERVOIR == "reservoir"
        assert TruncationStrategy.DIVERSE == "diverse"
        assert TruncationStrategy.STRATIFIED == "stratified"


class TestStatisticalSummary:
    """Tests for StatisticalSummary model."""

    def test_basic_summary(self) -> None:
        """Test basic summary creation."""
        summary = StatisticalSummary(
            count=10,
            min_value=1.0,
            max_value=10.0,
            mean=5.5,
        )
        assert summary.count == 10
        assert summary.min_value == 1.0
        assert summary.max_value == 10.0
        assert summary.mean == 5.5

    def test_range_property(self) -> None:
        """Test range calculation."""
        summary = StatisticalSummary(count=10, min_value=2.0, max_value=8.0, mean=5.0)
        assert summary.range == 6.0

    def test_optional_fields(self) -> None:
        """Test optional statistics."""
        summary = StatisticalSummary(
            count=100,
            min_value=0.0,
            max_value=100.0,
            mean=50.0,
            std_dev=29.15,
            median=50.0,
            sum_value=5000.0,
            percentiles={25: 25.0, 75: 75.0},
        )
        assert summary.std_dev == pytest.approx(29.15)
        assert summary.median == 50.0
        assert summary.percentiles[25] == 25.0


class TestTypeDistribution:
    """Tests for TypeDistribution model."""

    def test_basic_distribution(self) -> None:
        """Test basic type distribution."""
        dist = TypeDistribution(
            type_counts={"string": 5, "number": 3},
            total_items=8,
        )
        assert dist.type_counts["string"] == 5
        assert dist.total_items == 8

    def test_is_homogeneous_single_type(self) -> None:
        """Test homogeneous detection with single type."""
        dist = TypeDistribution(
            type_counts={"string": 10},
            total_items=10,
        )
        assert dist.is_homogeneous is True

    def test_is_homogeneous_multiple_types(self) -> None:
        """Test homogeneous detection with multiple types."""
        dist = TypeDistribution(
            type_counts={"string": 5, "number": 5},
            total_items=10,
        )
        assert dist.is_homogeneous is False

    def test_dominant_type(self) -> None:
        """Test dominant type detection."""
        dist = TypeDistribution(
            type_counts={"string": 7, "number": 2, "boolean": 1},
            total_items=10,
        )
        assert dist.dominant_type == "string"

    def test_type_percentages(self) -> None:
        """Test percentage calculation."""
        dist = TypeDistribution(
            type_counts={"string": 5, "number": 5},
            total_items=10,
        )
        percentages = dist.type_percentages
        assert percentages["string"] == pytest.approx(50.0)
        assert percentages["number"] == pytest.approx(50.0)


class TestTruncationResult:
    """Tests for TruncationResult model."""

    def test_basic_result(self) -> None:
        """Test basic truncation result."""
        result = TruncationResult(
            items=[1, 2, 3],
            original_count=10,
            kept_count=3,
            omitted_count=7,
            strategy=TruncationStrategy.UNIFORM,
            is_truncated=True,
        )
        assert len(result.items) == 3
        assert result.original_count == 10
        assert result.is_truncated is True

    def test_compression_ratio(self) -> None:
        """Test compression ratio calculation."""
        result = TruncationResult(
            items=[1, 2],
            original_count=10,
            kept_count=2,
            omitted_count=8,
            is_truncated=True,
        )
        assert result.compression_ratio == pytest.approx(5.0)

    def test_to_compressed_format(self) -> None:
        """Test conversion to compressed format."""
        result = TruncationResult(
            items=[1, 5, 10],
            original_count=100,
            kept_count=3,
            omitted_count=97,
            strategy=TruncationStrategy.UNIFORM,
            is_truncated=True,
        )
        compressed = result.to_compressed_format()

        assert compressed["_truncated"] is True
        assert compressed["_total_items"] == 100
        assert compressed["_kept_items"] == 3
        assert compressed["_strategy"] == "uniform"
        assert compressed["items"] == [1, 5, 10]

    def test_to_compressed_format_with_stats(self) -> None:
        """Test compressed format includes statistics."""
        stats = StatisticalSummary(
            count=100,
            min_value=1.0,
            max_value=100.0,
            mean=50.5,
            std_dev=28.9,
            median=50.0,
        )
        result = TruncationResult(
            items=[1, 50, 100],
            original_count=100,
            kept_count=3,
            omitted_count=97,
            statistical_summary=stats,
            is_truncated=True,
        )
        compressed = result.to_compressed_format()

        assert "_statistics" in compressed
        assert compressed["_statistics"]["min"] == 1.0
        assert compressed["_statistics"]["max"] == 100.0
        assert compressed["_statistics"]["mean"] == 50.5


class TestListTruncator:
    """Tests for ListTruncator class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        truncator = ListTruncator()
        assert truncator._default_strategy == TruncationStrategy.HEAD_TAIL
        assert truncator._include_statistics is True
        assert truncator._include_type_distribution is True

    def test_init_custom(self) -> None:
        """Test custom initialization."""
        truncator = ListTruncator(
            default_strategy=TruncationStrategy.UNIFORM,
            include_statistics=False,
            seed=42,
        )
        assert truncator._default_strategy == TruncationStrategy.UNIFORM
        assert truncator._include_statistics is False
        assert truncator._seed == 42

    def test_truncate_empty_list(self) -> None:
        """Test truncating empty list."""
        truncator = ListTruncator()
        result = truncator.truncate([], keep=5)

        assert result.items == []
        assert result.original_count == 0
        assert result.is_truncated is False

    def test_truncate_no_truncation_needed(self) -> None:
        """Test when list is already small enough."""
        truncator = ListTruncator()
        data = [1, 2, 3, 4, 5]
        result = truncator.truncate(data, keep=10)

        assert result.items == data
        assert result.is_truncated is False
        assert result.kept_count == 5

    def test_head_tail_strategy(self) -> None:
        """Test head/tail truncation strategy."""
        truncator = ListTruncator()
        data = list(range(20))
        result = truncator.truncate(data, keep=6, strategy=TruncationStrategy.HEAD_TAIL)

        assert result.is_truncated is True
        assert result.kept_count == 6
        # Should have first 3 and last 3
        assert result.items[:3] == [0, 1, 2]
        assert result.items[-3:] == [17, 18, 19]

    def test_uniform_strategy(self) -> None:
        """Test uniform sampling strategy."""
        truncator = ListTruncator()
        data = list(range(100))
        result = truncator.truncate(data, keep=10, strategy=TruncationStrategy.UNIFORM)

        assert result.is_truncated is True
        assert result.kept_count == 10
        # Should be evenly spaced
        assert 0 in result.items  # First
        assert 99 in result.items  # Last
        # Check spacing is roughly uniform
        indices = result.sample_indices
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        assert 9 <= avg_gap <= 12  # Roughly 11 apart

    def test_reservoir_strategy_deterministic(self) -> None:
        """Test reservoir sampling is deterministic with seed."""
        truncator = ListTruncator(seed=42)
        data = list(range(100))

        result1 = truncator.truncate(
            data, keep=10, strategy=TruncationStrategy.RESERVOIR
        )
        result2 = truncator.truncate(
            data, keep=10, strategy=TruncationStrategy.RESERVOIR
        )

        assert result1.items == result2.items

    def test_diverse_strategy_objects(self) -> None:
        """Test diverse strategy with objects."""
        truncator = ListTruncator()
        data = [
            {"type": "a", "value": 1},
            {"type": "a", "value": 2},
            {"type": "a", "value": 3},
            {"type": "b", "value": 4},
            {"type": "b", "value": 5},
            {"type": "c", "value": 6},
        ]
        result = truncator.truncate(data, keep=3, strategy=TruncationStrategy.DIVERSE)

        assert result.is_truncated is True
        assert result.kept_count == 3

    def test_stratified_strategy(self) -> None:
        """Test stratified sampling by type."""
        truncator = ListTruncator()
        data = [1, 2, 3, 4, 5, "a", "b", "c", True, False]  # Mixed types
        result = truncator.truncate(
            data, keep=5, strategy=TruncationStrategy.STRATIFIED
        )

        assert result.is_truncated is True
        assert result.kept_count == 5
        # Should have representation from different types
        types_in_result = {type(x).__name__ for x in result.items}
        assert len(types_in_result) >= 2

    def test_statistics_for_numeric_list(self) -> None:
        """Test statistics computation for numeric lists."""
        truncator = ListTruncator(include_statistics=True)
        data = list(range(1, 101))  # 1 to 100
        result = truncator.truncate(data, keep=10)

        assert result.statistical_summary is not None
        stats = result.statistical_summary
        assert stats.count == 100
        assert stats.min_value == 1.0
        assert stats.max_value == 100.0
        assert stats.mean == pytest.approx(50.5)
        assert stats.median == pytest.approx(50.5)

    def test_no_statistics_for_non_numeric(self) -> None:
        """Test no statistics for non-numeric lists."""
        truncator = ListTruncator(include_statistics=True)
        data = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]
        result = truncator.truncate(data, keep=5)

        assert result.statistical_summary is None

    def test_type_distribution_computed(self) -> None:
        """Test type distribution is computed."""
        truncator = ListTruncator(include_type_distribution=True)
        data = [1, 2, "a", "b", True, None] * 5  # 30 items
        result = truncator.truncate(data, keep=5)

        assert result.type_distribution is not None
        assert result.type_distribution.total_items == 30

    def test_auto_select_strategy_numeric(self) -> None:
        """Test auto-selection for numeric lists."""
        truncator = ListTruncator()
        data = list(range(100))
        strategy = truncator.auto_select_strategy(data)
        assert strategy == TruncationStrategy.UNIFORM

    def test_auto_select_strategy_heterogeneous(self) -> None:
        """Test auto-selection for mixed type lists."""
        truncator = ListTruncator()
        data = [1, "a", True, None, 2.5] * 10
        strategy = truncator.auto_select_strategy(data)
        assert strategy == TruncationStrategy.STRATIFIED

    def test_auto_select_strategy_objects_with_varying_keys(self) -> None:
        """Test auto-selection for objects with different structures."""
        truncator = ListTruncator()
        data = [
            {"a": 1},
            {"a": 1, "b": 2},
            {"a": 1, "c": 3},
        ] * 5
        strategy = truncator.auto_select_strategy(data)
        assert strategy == TruncationStrategy.DIVERSE

    def test_sample_indices_preserved(self) -> None:
        """Test that sample indices are tracked correctly."""
        truncator = ListTruncator()
        data = list(range(100))
        result = truncator.truncate(data, keep=10, strategy=TruncationStrategy.UNIFORM)

        assert len(result.sample_indices) == 10
        # Verify indices match items
        for i, idx in enumerate(result.sample_indices):
            assert result.items[i] == data[idx]


class TestToolResultCompressorTruncationIntegration:
    """Tests for ToolResultCompressor with smart truncation."""

    def test_compressor_uses_smart_truncation(self) -> None:
        """Test compressor uses smart truncation by default."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=3,
            use_smart_truncation=True,
        )

        data = list(range(20))
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        assert "truncation_strategy" in result.metadata

    def test_compressor_smart_truncation_with_statistics(self) -> None:
        """Test compressor includes statistics for numeric lists."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=3,
            use_smart_truncation=True,
        )

        data = list(range(1, 101))
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        assert result.metadata.get("has_statistics") is True
        # Check compressed content has statistics
        assert "_statistics" in result.compressed_content

    def test_compressor_disable_smart_truncation(self) -> None:
        """Test disabling smart truncation falls back to simple."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=4,
            use_smart_truncation=False,
        )

        data = list(range(20))
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        # Should use simple format
        assert "_showing" in result.compressed_content
        assert "truncation_strategy" not in result.metadata

    def test_compressor_explicit_strategy(self) -> None:
        """Test compressor with explicit truncation strategy."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=5,
            use_smart_truncation=True,
            truncation_strategy=TruncationStrategy.UNIFORM,
        )

        data = list(range(100))
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("truncation_strategy") == "uniform"

    def test_compressor_nested_list_truncation(self) -> None:
        """Test truncation of nested lists."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=3,
        )

        data = {
            "results": list(range(20)),
            "metadata": {"count": 20},
        }
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        # The nested list should be truncated
        assert result.compressed_content["results"]["_truncated"] is True

    def test_compressor_preserves_small_lists(self) -> None:
        """Test that small lists are not truncated."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=10,
            list_keep_items=5,
        )

        data = [1, 2, 3, 4, 5]
        result = compressor.compress("test_tool", data)

        # Should not be truncated
        assert result.compressed_content == [1, 2, 3, 4, 5]
        assert result.metadata.get("list_truncated") is None

    def test_compressor_mixed_type_list(self) -> None:
        """Test truncation of mixed type lists."""
        compressor = ToolResultCompressor(
            list_truncate_threshold=5,
            list_keep_items=4,
            use_smart_truncation=True,
        )

        data = [1, "a", 2, "b", 3, "c", 4, "d", 5, "e"] * 2
        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        # Should have type distribution info
        assert "type_distribution" in result.metadata
