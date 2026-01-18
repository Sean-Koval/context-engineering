"""Tests for ToolResultCompressor."""

from __future__ import annotations

from context_tools import CompressionResult, ToolResultCompressor


class TestToolResultCompressor:
    """Tests for ToolResultCompressor class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        compressor = ToolResultCompressor()
        assert compressor._max_tokens == 2000
        assert compressor._list_threshold == 10
        assert compressor._list_keep == 5

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        compressor = ToolResultCompressor(
            max_tokens=1000,
            list_truncate_threshold=5,
            list_keep_items=3,
        )
        assert compressor._max_tokens == 1000
        assert compressor._list_threshold == 5
        assert compressor._list_keep == 3

    def test_compress_simple_dict(self) -> None:
        """Test compression of a simple dictionary."""
        compressor = ToolResultCompressor()
        data = {"name": "test", "value": 42}

        result = compressor.compress("test_tool", data)

        assert isinstance(result, CompressionResult)
        assert result.compressed_content == data
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0

    def test_filter_low_value_fields(self) -> None:
        """Test that low-value fields are filtered out."""
        compressor = ToolResultCompressor()
        data = {
            "name": "test",
            "_metadata": {"internal": True},
            "debug": {"verbose": True},
            "trace": "abc123",
        }

        result = compressor.compress("test_tool", data)

        assert "name" in result.compressed_content
        assert "_metadata" not in result.compressed_content
        assert "debug" not in result.compressed_content
        assert "trace" not in result.compressed_content

    def test_filter_underscore_fields(self) -> None:
        """Test that underscore-prefixed fields are filtered except _type and _id."""
        compressor = ToolResultCompressor()
        data = {
            "name": "test",
            "_type": "important",
            "_id": "123",
            "_internal": "should be removed",
            "_raw": "should be removed",
        }

        result = compressor.compress("test_tool", data)

        assert result.compressed_content["name"] == "test"
        assert result.compressed_content["_type"] == "important"
        assert result.compressed_content["_id"] == "123"
        assert "_internal" not in result.compressed_content
        assert "_raw" not in result.compressed_content

    def test_truncate_long_list(self) -> None:
        """Test truncation of lists exceeding threshold."""
        compressor = ToolResultCompressor(list_truncate_threshold=5, list_keep_items=4)
        data = list(range(20))

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is True
        assert result.compressed_content["_truncated"] is True
        assert result.compressed_content["_total_items"] == 20
        # Should have first 2 + last 2 + 1 summary = 5 items in "items"
        assert len(result.compressed_content["items"]) == 5

    def test_no_truncation_below_threshold(self) -> None:
        """Test that lists below threshold are not truncated."""
        compressor = ToolResultCompressor(list_truncate_threshold=10)
        data = list(range(5))

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("list_truncated") is None
        assert result.compressed_content == [0, 1, 2, 3, 4]

    def test_schema_extraction_homogeneous_list(self) -> None:
        """Test schema extraction for list of similar dicts."""
        compressor = ToolResultCompressor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b", "value": 2},
            {"name": "c", "value": 3},
        ]

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("schema_extracted") is True
        assert "_schema" in result.compressed_content
        assert "_keys" in result.compressed_content
        assert "_values" in result.compressed_content
        assert result.compressed_content["_keys"] == ["name", "value"]
        assert len(result.compressed_content["_values"]) == 3

    def test_no_schema_extraction_heterogeneous_list(self) -> None:
        """Test that schema extraction doesn't apply to heterogeneous lists."""
        compressor = ToolResultCompressor()
        data = [
            {"name": "a", "value": 1},
            {"name": "b", "other": 2},  # Different keys
            {"name": "c", "value": 3},
        ]

        result = compressor.compress("test_tool", data)

        assert result.metadata.get("schema_extracted") is None
        # Should be a list, not schema-extracted
        assert isinstance(result.compressed_content, list)

    def test_tool_specific_compression_read_file(self) -> None:
        """Test tool-specific compression for file reads."""
        compressor = ToolResultCompressor()
        code = "line1\n\n\n\nline2\n\n\nline3"  # Multiple blank lines

        result = compressor.compress("read_file", code)

        # Consecutive blank lines should be collapsed
        assert "\n\n\n" not in result.compressed_content
        assert "line1" in result.compressed_content
        assert "line2" in result.compressed_content

    def test_tool_specific_compression_search(self) -> None:
        """Test tool-specific compression for search results."""
        compressor = ToolResultCompressor()
        data = [
            {"path": "/a.py", "line": 10, "match": "foo", "extra": "remove me"},
            {"path": "/b.py", "line": 20, "match": "bar", "internal": "gone"},
        ]

        result = compressor.compress("search", data)

        # Should keep essential fields, remove extras
        for item in result.compressed_content:
            assert "path" in item
            assert "line" in item
            assert "match" in item
            assert "extra" not in item
            assert "internal" not in item

    def test_tool_specific_compression_web_fetch(self) -> None:
        """Test tool-specific compression for web content."""
        compressor = ToolResultCompressor()
        html = "<html><body><p>Hello</p><p>World</p></body></html>"

        result = compressor.compress("web_fetch", html)

        # HTML tags should be stripped
        assert "<html>" not in result.compressed_content
        assert "<body>" not in result.compressed_content
        assert "Hello" in result.compressed_content
        assert "World" in result.compressed_content

    def test_compression_ratio_calculated(self) -> None:
        """Test that compression ratio is calculated correctly."""
        compressor = ToolResultCompressor()
        # Create data that will compress significantly
        data = {
            "name": "test",
            "_metadata": {"a": 1, "b": 2, "c": 3},
            "debug": {"x": "y"},
            "trace": "long trace string here",
        }

        result = compressor.compress("test_tool", data)

        # Should have positive ratio since we removed fields
        assert result.compression_ratio >= 1.0
        assert result.tokens_saved >= 0

    def test_custom_token_counter(self) -> None:
        """Test using a custom token counter function."""
        compressor = ToolResultCompressor()

        def custom_counter(text: str) -> int:
            return len(text)  # Simple char count

        result = compressor.compress(
            "test_tool", {"a": 1}, token_counter=custom_counter
        )

        # With char counting, tokens should equal string length
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0

    def test_strategy_used_tracking(self) -> None:
        """Test that strategies used are tracked."""
        compressor = ToolResultCompressor()

        # Dict triggers field_filter
        result = compressor.compress("test_tool", {"name": "test"})
        assert "field_filter" in result.strategy_used

        # Long list triggers list_truncation
        compressor2 = ToolResultCompressor(list_truncate_threshold=3)
        result2 = compressor2.compress("test_tool", list(range(10)))
        assert "list_truncation" in result2.strategy_used

    def test_is_reversible_flag(self) -> None:
        """Test that is_reversible is set correctly."""
        compressor = ToolResultCompressor()

        # Schema extraction is not reversible
        data = [{"a": 1}, {"a": 2}, {"a": 3}]
        result = compressor.compress("test_tool", data)
        assert result.is_reversible is False

        # Simple filtering is reversible
        result2 = compressor.compress("test_tool", {"name": "test"})
        assert result2.is_reversible is True

    def test_nested_dict_filtering(self) -> None:
        """Test that filtering works on nested dicts."""
        compressor = ToolResultCompressor()
        data = {
            "outer": {
                "inner": {"name": "test", "_raw": "remove", "debug": "remove"},
                "_metadata": "remove",
            }
        }

        result = compressor.compress("test_tool", data)

        assert "outer" in result.compressed_content
        assert "inner" in result.compressed_content["outer"]
        assert "name" in result.compressed_content["outer"]["inner"]
        assert "_raw" not in result.compressed_content["outer"]["inner"]
        assert "_metadata" not in result.compressed_content["outer"]

    def test_empty_input(self) -> None:
        """Test handling of empty inputs."""
        compressor = ToolResultCompressor()

        result_dict = compressor.compress("test_tool", {})
        assert result_dict.compressed_content == {}

        result_list = compressor.compress("test_tool", [])
        assert result_list.compressed_content == []

        result_str = compressor.compress("test_tool", "")
        assert result_str.compressed_content == ""
