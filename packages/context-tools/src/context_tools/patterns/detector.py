"""Tool usage pattern detection and learning.

Provides pattern detection for tool call sequences to enable:
- Sequence pattern discovery (A -> B -> C)
- Next-tool prediction based on learned transitions
- Parameter pattern learning for common values
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from context_tools.types import (
    Antipattern,
    AntipatternType,
    ToolCallSignature,
    ToolPattern,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class ToolSequence(BaseModel):
    """A detected sequence of tool calls.

    Attributes:
        tools: Ordered list of tool names in the sequence
        frequency: Number of times this sequence was observed
        avg_gap_ms: Average time between consecutive tools in sequence
    """

    tools: list[str] = Field(min_length=2)
    frequency: int = Field(default=1, ge=1)
    avg_gap_ms: float = Field(default=0.0, ge=0.0)


class ParameterStats(BaseModel):
    """Statistics for a parameter's observed values.

    Attributes:
        param_name: Name of the parameter
        value_type: Python type name of values
        value_counts: Frequency of each observed value (for discrete values)
        total_observations: Total number of observations
    """

    param_name: str
    value_type: str = "str"
    value_counts: dict[str, int] = Field(default_factory=dict)
    total_observations: int = Field(default=0, ge=0)

    def record_value(self, value: Any) -> None:
        """Record an observed value."""
        self.total_observations += 1
        # Only track discrete values that aren't too long
        if isinstance(value, (str, int, bool)) and len(str(value)) < 100:
            str_value = str(value)
            self.value_counts[str_value] = self.value_counts.get(str_value, 0) + 1

    def most_common_value(self) -> str | None:
        """Get the most frequently observed value."""
        if not self.value_counts:
            return None
        return max(self.value_counts.items(), key=lambda x: x[1])[0]


class UsageStats(BaseModel):
    """Statistics about tool usage patterns.

    Attributes:
        history_size: Number of tool calls in history
        unique_tools: Number of distinct tools observed
        tool_frequencies: Count of calls per tool
        sequence_count: Number of detected sequences
        transition_count: Number of observed transitions
    """

    history_size: int = Field(default=0, ge=0)
    unique_tools: int = Field(default=0, ge=0)
    tool_frequencies: dict[str, int] = Field(default_factory=dict)
    sequence_count: int = Field(default=0, ge=0)
    transition_count: int = Field(default=0, ge=0)


class ToolUsagePatterns:
    """Learn and detect patterns from tool usage history.

    Tracks tool call sequences to enable:
    - Sequence pattern detection (common tool chains)
    - Next-tool prediction based on transition probabilities
    - Parameter pattern learning for argument prediction

    Example:
        >>> patterns = ToolUsagePatterns()
        >>> # Record tool calls as they happen
        >>> sig1 = ToolCallSignature(tool_name="search", arguments={"query": "foo"})
        >>> patterns.record(sig1)
        >>> sig2 = ToolCallSignature(tool_name="read_file", arguments={"path": "/a.py"})
        >>> patterns.record(sig2)
        >>> # Predict next tool
        >>> predictions = patterns.predict_next_tool("search")
        >>> print(predictions)  # [("read_file", 1.0)]

    Attributes:
        window_size: Maximum history size to maintain
        min_pattern_frequency: Minimum occurrences to consider a pattern
        max_sequence_length: Maximum length of sequences to detect
    """

    def __init__(
        self,
        window_size: int = 10000,
        min_pattern_frequency: int = 3,
        max_sequence_length: int = 5,
    ) -> None:
        """Initialize pattern detector.

        Args:
            window_size: Maximum number of tool calls to keep in history
            min_pattern_frequency: Minimum frequency for pattern detection
            max_sequence_length: Maximum sequence length to detect
        """
        self._history: list[ToolCallSignature] = []
        self._window_size = window_size
        self._min_frequency = min_pattern_frequency
        self._max_sequence_length = max_sequence_length

        # Transition matrix: from_tool -> to_tool -> count
        self._transitions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Parameter patterns: tool_name -> param_name -> ParameterStats
        self._parameter_patterns: dict[str, dict[str, ParameterStats]] = defaultdict(
            dict
        )

        # Cached detected patterns
        self._cached_patterns: list[ToolPattern] = []
        self._patterns_dirty = True

    def record(self, signature: ToolCallSignature) -> None:
        """Record a tool call for pattern learning.

        Updates transition matrix and parameter patterns based on
        the new tool call.

        Args:
            signature: The tool call signature to record
        """
        # Add timestamp if not present
        if not hasattr(signature, "timestamp"):
            signature = ToolCallSignature(
                tool_name=signature.tool_name,
                arguments=signature.arguments,
                context_hash=signature.context_hash,
            )

        self._history.append(signature)

        # Update transition matrix
        if len(self._history) >= 2:
            prev_tool = self._history[-2].tool_name
            curr_tool = signature.tool_name
            self._transitions[prev_tool][curr_tool] += 1

        # Update parameter patterns
        self._update_parameter_patterns(signature)

        # Mark patterns as needing refresh
        self._patterns_dirty = True

        # Trim history if needed
        if len(self._history) > self._window_size:
            self._history = self._history[-self._window_size :]

    def _update_parameter_patterns(self, signature: ToolCallSignature) -> None:
        """Update parameter statistics for a tool call."""
        tool = signature.tool_name

        for param_name, value in signature.arguments.items():
            if param_name not in self._parameter_patterns[tool]:
                self._parameter_patterns[tool][param_name] = ParameterStats(
                    param_name=param_name,
                    value_type=type(value).__name__,
                )
            self._parameter_patterns[tool][param_name].record_value(value)

    def detect_sequences(self, max_length: int = 3) -> list[ToolSequence]:
        """Detect common tool call sequences.

        Scans history for recurring sequences of tool calls that appear
        at least min_pattern_frequency times.

        Args:
            max_length: Maximum sequence length to detect (2 to max_sequence_length)

        Returns:
            List of detected sequences, sorted by frequency (descending)
        """
        max_length = min(max_length, self._max_sequence_length)
        sequence_data: dict[tuple[str, ...], list[float]] = defaultdict(list)

        # Sliding window over history
        for length in range(2, max_length + 1):
            for i in range(len(self._history) - length + 1):
                window = self._history[i : i + length]
                key = tuple(sig.tool_name for sig in window)

                # Calculate time gap (use 0 if timestamps not available)
                time_gap_ms = 0.0
                sequence_data[key].append(time_gap_ms)

        # Filter by frequency and build results
        results: list[ToolSequence] = []
        for key, gaps in sequence_data.items():
            if len(gaps) >= self._min_frequency:
                avg_gap = sum(gaps) / len(gaps) if gaps else 0.0
                results.append(
                    ToolSequence(
                        tools=list(key),
                        frequency=len(gaps),
                        avg_gap_ms=avg_gap,
                    )
                )

        return sorted(results, key=lambda s: s.frequency, reverse=True)

    def predict_next_tool(
        self,
        current_tool: str,
        context_tools: Sequence[str] | None = None,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Predict most likely next tool based on transition history.

        Uses the transition matrix to calculate probabilities for
        what tool is likely to be called next.

        Args:
            current_tool: The tool that was just called
            context_tools: Recent tools for context boosting (optional)
            top_k: Number of predictions to return

        Returns:
            List of (tool_name, probability) tuples, sorted by probability
        """
        if current_tool not in self._transitions:
            return []

        transitions = self._transitions[current_tool]
        total = sum(transitions.values())

        if total == 0:
            return []

        # Calculate base probabilities
        predictions: list[tuple[str, float]] = [
            (tool, count / total) for tool, count in transitions.items()
        ]

        # Apply context boost if recent tools provided
        if context_tools:
            recent_set = set(list(context_tools)[-5:])
            boosted: list[tuple[str, float]] = []
            for tool, prob in predictions:
                boost = 1.2 if tool in recent_set else 1.0
                boosted.append((tool, prob * boost))
            predictions = boosted

        # Sort by probability and return top-k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def get_parameter_pattern(
        self,
        tool_name: str,
        param_name: str,
    ) -> ParameterStats | None:
        """Get learned parameter statistics for a tool parameter.

        Args:
            tool_name: Name of the tool
            param_name: Name of the parameter

        Returns:
            ParameterStats if available, None otherwise
        """
        if tool_name not in self._parameter_patterns:
            return None
        return self._parameter_patterns[tool_name].get(param_name)

    def predict_arguments(
        self,
        tool_name: str,
        partial_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Predict likely arguments for a tool based on patterns.

        Uses learned parameter patterns to suggest argument values.

        Args:
            tool_name: Name of the tool
            partial_args: Already-known arguments (won't be overwritten)

        Returns:
            Dictionary of predicted argument values
        """
        predicted: dict[str, Any] = {}
        partial = partial_args or {}

        if tool_name in self._parameter_patterns:
            for param_name, stats in self._parameter_patterns[tool_name].items():
                # Don't override existing args
                if param_name in partial:
                    continue

                most_common = stats.most_common_value()
                if most_common is not None:
                    # Try to parse back to original type
                    if stats.value_type == "int":
                        try:
                            predicted[param_name] = int(most_common)
                        except ValueError:
                            predicted[param_name] = most_common
                    elif stats.value_type == "bool":
                        predicted[param_name] = most_common.lower() == "true"
                    else:
                        predicted[param_name] = most_common

        return predicted

    def get_transition_probability(self, from_tool: str, to_tool: str) -> float:
        """Get the probability of transitioning between two tools.

        Args:
            from_tool: Source tool name
            to_tool: Target tool name

        Returns:
            Probability (0.0 to 1.0) of the transition
        """
        if from_tool not in self._transitions:
            return 0.0

        transitions = self._transitions[from_tool]
        total = sum(transitions.values())

        if total == 0:
            return 0.0

        return transitions.get(to_tool, 0) / total

    def get_tool_frequency(self, tool_name: str) -> int:
        """Get the total number of times a tool was called.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of recorded calls
        """
        return sum(1 for sig in self._history if sig.tool_name == tool_name)

    def get_stats(self) -> UsageStats:
        """Get summary statistics about tool usage.

        Returns:
            UsageStats with history and pattern information
        """
        tool_counts: dict[str, int] = defaultdict(int)
        for sig in self._history:
            tool_counts[sig.tool_name] += 1

        transition_count = sum(
            sum(targets.values()) for targets in self._transitions.values()
        )

        return UsageStats(
            history_size=len(self._history),
            unique_tools=len(tool_counts),
            tool_frequencies=dict(tool_counts),
            sequence_count=len(self.detect_sequences()),
            transition_count=transition_count,
        )

    def clear(self) -> None:
        """Clear all recorded history and patterns."""
        self._history.clear()
        self._transitions.clear()
        self._parameter_patterns.clear()
        self._cached_patterns.clear()
        self._patterns_dirty = True

    def to_patterns(self) -> list[ToolPattern]:
        """Convert detected sequences to ToolPattern objects.

        Returns:
            List of ToolPattern objects for detected sequences
        """
        if not self._patterns_dirty and self._cached_patterns:
            return self._cached_patterns

        sequences = self.detect_sequences(max_length=self._max_sequence_length)
        patterns: list[ToolPattern] = []

        for seq in sequences:
            confidence = min(1.0, seq.frequency / 10)  # Saturates at 10
            patterns.append(
                ToolPattern(
                    sequence=seq.tools,
                    frequency=seq.frequency,
                    confidence=confidence,
                    avg_gap_ms=seq.avg_gap_ms,
                )
            )

        self._cached_patterns = patterns
        self._patterns_dirty = False
        return patterns

    def detect_antipatterns(self) -> list[Antipattern]:
        """Detect inefficient tool usage patterns.

        Scans history for patterns that indicate inefficient or wasteful
        tool usage, such as repeated identical calls or unused results.

        Returns:
            List of detected antipatterns with suggestions for improvement
        """
        antipatterns: list[Antipattern] = []

        # Detect repeated identical calls
        antipatterns.extend(self._detect_repeated_calls())

        # Detect unused search results
        antipatterns.extend(self._detect_unused_results())

        # Detect multiple reads of the same file
        antipatterns.extend(self._detect_multi_reads())

        return antipatterns

    def _detect_repeated_calls(self) -> list[Antipattern]:
        """Detect repeated identical tool calls.

        Finds cases where the same tool is called with identical arguments
        multiple times, which usually indicates missing caching.
        """
        import json

        call_counts: dict[tuple[str, str], int] = defaultdict(int)

        for sig in self._history:
            # Create a hashable key from tool name and sorted arguments
            args_json = json.dumps(sig.arguments, sort_keys=True)
            key = (sig.tool_name, args_json)
            call_counts[key] += 1

        patterns: list[Antipattern] = []
        for (tool, _), count in call_counts.items():
            if count >= 3:  # Same call 3+ times suggests missing caching
                desc = f"Tool '{tool}' called {count} times with identical args"
                patterns.append(
                    Antipattern(
                        type=AntipatternType.REDUNDANT_CALL,
                        tool_name=tool,
                        description=desc,
                        occurrences=count,
                        tokens_wasted=(count - 1) * 100,
                        suggestion="Consider caching results for repeated calls",
                    )
                )

        return patterns

    def _detect_unused_results(self) -> list[Antipattern]:
        """Detect search/query results that appear unused.

        Finds cases where a search or query tool is called but the next
        tool is not a read or follow-up action on the results.
        """
        search_tools = {"search", "grep", "find", "glob", "query", "list"}
        follow_up_tools = {"read", "read_file", "get", "fetch", "open"}

        patterns: list[Antipattern] = []

        for i in range(len(self._history) - 1):
            curr = self._history[i]
            next_call = self._history[i + 1]

            # Check if current is a search-type tool
            curr_lower = curr.tool_name.lower()
            if any(s in curr_lower for s in search_tools):
                # Check if next is NOT a follow-up tool
                next_lower = next_call.tool_name.lower()
                if not any(f in next_lower for f in follow_up_tools):
                    desc = (
                        f"Search '{curr.tool_name}' followed by "
                        f"'{next_call.tool_name}' instead of reading results"
                    )
                    patterns.append(
                        Antipattern(
                            type=AntipatternType.UNUSED_RESULT,
                            tool_name=curr.tool_name,
                            description=desc,
                            occurrences=1,
                            tokens_wasted=50,
                            suggestion="Use search results before other operations",
                        )
                    )

        return patterns

    def _detect_multi_reads(self) -> list[Antipattern]:
        """Detect multiple reads of the same file within a short window.

        Finds cases where the same file is read multiple times in close
        succession, which usually indicates the content should be cached.
        """
        read_tools = {"read", "read_file", "cat", "get_file"}
        path_params = {"path", "file_path", "filename", "file"}

        # Track file reads: path -> list of indices
        file_reads: dict[str, list[int]] = defaultdict(list)

        for i, sig in enumerate(self._history):
            tool_lower = sig.tool_name.lower()
            if any(r in tool_lower for r in read_tools):
                # Extract path from arguments
                for param in path_params:
                    if param in sig.arguments:
                        path = str(sig.arguments[param])
                        file_reads[path].append(i)
                        break

        patterns: list[Antipattern] = []
        for path, indices in file_reads.items():
            if len(indices) < 2:
                continue

            # Check for reads close together (within 10 tool calls)
            for i in range(len(indices) - 1):
                gap = indices[i + 1] - indices[i]
                if gap <= 10:
                    desc = f"File '{path}' read {len(indices)}x within {gap} calls"
                    patterns.append(
                        Antipattern(
                            type=AntipatternType.REDUNDANT_CALL,
                            tool_name="read_file",
                            description=desc,
                            occurrences=len(indices),
                            tokens_wasted=(len(indices) - 1) * 200,
                            suggestion="Cache file contents to avoid re-reading",
                        )
                    )
                    break  # Only report once per file

        return patterns

    @property
    def history_size(self) -> int:
        """Get current history size."""
        return len(self._history)

    @property
    def transition_count(self) -> int:
        """Get total number of recorded transitions."""
        return sum(sum(targets.values()) for targets in self._transitions.values())
