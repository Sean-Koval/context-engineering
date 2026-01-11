"""Tests for compression types."""

from __future__ import annotations

from uuid import uuid4

import pytest

from context_compression.types import (
    CompressionPlan,
    CompressionResult,
    CompressionTier,
    PipelineConfig,
    PreservationRule,
)


class TestCompressionTier:
    """Tests for CompressionTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert CompressionTier.LOSSLESS.value == "lossless"
        assert CompressionTier.COMPACTION.value == "compaction"
        assert CompressionTier.SUMMARIZATION.value == "summarization"

    def test_tier_from_string(self):
        """Test creating tier from string."""
        assert CompressionTier("lossless") == CompressionTier.LOSSLESS
        assert CompressionTier("compaction") == CompressionTier.COMPACTION
        assert CompressionTier("summarization") == CompressionTier.SUMMARIZATION


class TestCompressionResult:
    """Tests for CompressionResult model."""

    def test_basic_result(self):
        """Test creating a basic result."""
        result = CompressionResult(
            success=True,
            strategy_name="test_strategy",
            tier=CompressionTier.LOSSLESS,
            original_tokens=1000,
            compressed_tokens=500,
            tokens_saved=500,
        )
        assert result.success is True
        assert result.strategy_name == "test_strategy"
        assert result.tier == CompressionTier.LOSSLESS
        assert result.original_tokens == 1000
        assert result.compressed_tokens == 500
        assert result.tokens_saved == 500

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        result = CompressionResult(
            success=True,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=1000,
            compressed_tokens=250,
            tokens_saved=750,
        )
        assert result.compression_ratio == 4.0

    def test_compression_ratio_zero_compressed(self):
        """Test compression ratio when compressed is zero."""
        result = CompressionResult(
            success=True,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=1000,
            compressed_tokens=0,
            tokens_saved=1000,
        )
        assert result.compression_ratio == float("inf")

    def test_savings_percent(self):
        """Test savings percentage calculation."""
        result = CompressionResult(
            success=True,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=1000,
            compressed_tokens=500,
            tokens_saved=500,
        )
        assert result.savings_percent == 50.0

    def test_savings_percent_zero_original(self):
        """Test savings percent when original is zero."""
        result = CompressionResult(
            success=True,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=0,
            compressed_tokens=0,
            tokens_saved=0,
        )
        assert result.savings_percent == 0.0

    def test_failed_result_with_error(self):
        """Test creating a failed result with error message."""
        result = CompressionResult(
            success=False,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=0,
            compressed_tokens=0,
            tokens_saved=0,
            error_message="Something went wrong",
        )
        assert result.success is False
        assert result.error_message == "Something went wrong"

    def test_result_with_manifest_id(self):
        """Test result with manifest ID."""
        manifest_id = uuid4()
        result = CompressionResult(
            success=True,
            strategy_name="test",
            tier=CompressionTier.LOSSLESS,
            original_tokens=100,
            compressed_tokens=50,
            tokens_saved=50,
            manifest_id=manifest_id,
        )
        assert result.manifest_id == manifest_id


class TestCompressionPlan:
    """Tests for CompressionPlan model."""

    def test_basic_plan(self):
        """Test creating a basic plan."""
        plan = CompressionPlan(
            strategies=["strategy1", "strategy2"],
            target_tokens=5000,
            estimated_savings=3000,
        )
        assert plan.strategies == ["strategy1", "strategy2"]
        assert plan.target_tokens == 5000
        assert plan.estimated_savings == 3000

    def test_plan_with_nodes(self):
        """Test plan with affected and preserved nodes."""
        node_ids = [uuid4(), uuid4()]
        preserved_ids = [uuid4()]

        plan = CompressionPlan(
            strategies=["test"],
            nodes_affected=node_ids,
            preserved_nodes=preserved_ids,
            preservations=["recent_n: 5 nodes"],
        )
        assert len(plan.nodes_affected) == 2
        assert len(plan.preserved_nodes) == 1
        assert "recent_n: 5 nodes" in plan.preservations

    def test_empty_plan(self):
        """Test creating an empty plan."""
        plan = CompressionPlan()
        assert plan.strategies == []
        assert plan.target_tokens is None
        assert plan.estimated_savings == 0


class TestPreservationRule:
    """Tests for PreservationRule model."""

    def test_basic_rule(self):
        """Test creating a basic rule."""
        rule = PreservationRule(
            name="test_rule",
            description="A test rule",
            priority=50,
        )
        assert rule.name == "test_rule"
        assert rule.description == "A test rule"
        assert rule.priority == 50

    def test_rule_with_node_types(self):
        """Test rule with node type filter."""
        rule = PreservationRule(
            name="system_only",
            node_types=["system", "message"],
        )
        assert rule.node_types == ["system", "message"]

    def test_rule_node_types_from_string(self):
        """Test node_types validator accepts string."""
        rule = PreservationRule(
            name="single_type",
            node_types="system",  # type: ignore
        )
        assert rule.node_types == ["system"]

    def test_rule_with_importance(self):
        """Test rule with importance threshold."""
        rule = PreservationRule(
            name="high_importance",
            min_importance=0.8,
        )
        assert rule.min_importance == 0.8

    def test_rule_with_age(self):
        """Test rule with age limit."""
        rule = PreservationRule(
            name="recent",
            max_age_seconds=300,
        )
        assert rule.max_age_seconds == 300

    def test_rule_with_tags(self):
        """Test rule with required tags."""
        rule = PreservationRule(
            name="tagged",
            required_tags={"important", "keep"},
        )
        assert rule.required_tags == {"important", "keep"}

    def test_rule_with_entities(self):
        """Test rule with entity IDs."""
        rule = PreservationRule(
            name="entity_rule",
            entity_ids=["entity1", "entity2"],
        )
        assert rule.entity_ids == ["entity1", "entity2"]

    def test_pinned_rule(self):
        """Test pinned rule."""
        rule = PreservationRule(
            name="pinned",
            pinned=True,
        )
        assert rule.pinned is True

    def test_rule_validation_importance_range(self):
        """Test importance must be in valid range."""
        with pytest.raises(ValueError):
            PreservationRule(name="invalid", min_importance=1.5)

        with pytest.raises(ValueError):
            PreservationRule(name="invalid", min_importance=-0.1)

    def test_rule_validation_age_positive(self):
        """Test age must be non-negative."""
        with pytest.raises(ValueError):
            PreservationRule(name="invalid", max_age_seconds=-1)


class TestPipelineConfig:
    """Tests for PipelineConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.preserve_recent_n == 10
        assert config.enable_recovery is True
        assert config.max_iterations == 10
        assert config.stop_on_error is False
        assert config.min_savings_threshold == 100

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            preserve_recent_n=20,
            enable_recovery=False,
            max_iterations=5,
            stop_on_error=True,
            min_savings_threshold=50,
        )
        assert config.preserve_recent_n == 20
        assert config.enable_recovery is False
        assert config.max_iterations == 5
        assert config.stop_on_error is True
        assert config.min_savings_threshold == 50

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            PipelineConfig(preserve_recent_n=-1)

        with pytest.raises(ValueError):
            PipelineConfig(max_iterations=0)
