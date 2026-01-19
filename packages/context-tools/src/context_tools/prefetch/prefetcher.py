"""Predictive tool prefetching for latency reduction.

Provides speculative execution of predicted tool calls based on
learned usage patterns to reduce user-perceived latency.

Example:
    >>> async def execute_tool(name: str, args: dict) -> Any:
    ...     return {"result": "data"}
    >>> prefetcher = ToolPrefetcher(patterns, cache, execute_tool)
    >>> # After a tool completes, trigger prefetching
    >>> results = await prefetcher.on_tool_complete(signature, result)
    >>> # Later, check if prefetch is ready
    >>> cached = await prefetcher.await_prefetch(next_signature, timeout_ms=100)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from context_tools.types import (
    PrefetchCandidate,
    PrefetchResult,
    PrefetchStats,
    ToolCallSignature,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from context_tools.cache.cache import ToolCallCache
    from context_tools.patterns.detector import ToolUsagePatterns


class ToolPrefetcher:
    """Predictive tool execution for latency reduction.

    Uses pattern detection to predict likely next tools and
    speculatively executes them in the background, caching results
    for immediate use when actually requested.

    The prefetcher monitors tool call completions, predicts what
    tools are likely to be called next based on learned patterns,
    and proactively executes those predictions in background tasks.

    Example:
        >>> patterns = ToolUsagePatterns()
        >>> cache = ToolCallCache()
        >>> async def executor(name: str, args: dict) -> Any:
        ...     # Your tool execution logic
        ...     return await execute_actual_tool(name, args)
        >>> prefetcher = ToolPrefetcher(
        ...     patterns=patterns,
        ...     cache=cache,
        ...     tool_executor=executor,
        ...     min_confidence=0.5,
        ... )
        >>> # Hook into tool completions
        >>> result = await actual_tool_call()
        >>> await prefetcher.on_tool_complete(signature, result)

    Attributes:
        stats: Prefetch performance statistics
    """

    def __init__(
        self,
        patterns: ToolUsagePatterns,
        cache: ToolCallCache,
        tool_executor: Callable[[str, dict[str, Any]], Awaitable[Any]],
        max_concurrent_prefetch: int = 3,
        min_confidence: float = 0.3,
        default_latency_ms: float = 100.0,
    ) -> None:
        """Initialize ToolPrefetcher.

        Args:
            patterns: Pattern detector for predicting next tools
            cache: Cache for storing prefetched results
            tool_executor: Async function to execute tools (name, args) -> result
            max_concurrent_prefetch: Maximum concurrent prefetch operations
            min_confidence: Minimum prediction confidence to trigger prefetch
            default_latency_ms: Default expected latency for prefetch decisions
        """
        self._patterns = patterns
        self._cache = cache
        self._executor = tool_executor
        self._max_concurrent = max_concurrent_prefetch
        self._min_confidence = min_confidence
        self._default_latency_ms = default_latency_ms

        # Pending prefetch tasks: task_key -> (Task, start_time, signature)
        self._pending: dict[
            str, tuple[asyncio.Task[Any], float, ToolCallSignature]
        ] = {}

        # Statistics
        self._stats = PrefetchStats()

        # Lock for thread safety
        self._lock = asyncio.Lock()

    @property
    def stats(self) -> PrefetchStats:
        """Get prefetch statistics."""
        self._stats.pending_count = len(self._pending)
        return self._stats

    async def on_tool_complete(
        self,
        signature: ToolCallSignature,
        result: Any,
    ) -> list[PrefetchResult]:
        """Called after a tool completes to trigger prefetching.

        Records the tool call in patterns and initiates prefetch
        operations for predicted next tools.

        Args:
            signature: The completed tool call signature
            result: The result from the tool execution

        Returns:
            List of prefetch operations that were started
        """
        # Record the call in patterns
        self._patterns.record(signature)

        # Get predictions for next tools
        predictions = self._patterns.predict_next_tool(
            signature.tool_name,
            top_k=self._max_concurrent,
        )

        prefetch_results: list[PrefetchResult] = []

        async with self._lock:
            # Clean up completed tasks
            await self._cleanup_completed()

            for tool_name, confidence in predictions:
                # Skip if confidence too low
                if confidence < self._min_confidence:
                    continue

                # Skip if we're at max concurrent prefetches
                if len(self._pending) >= self._max_concurrent:
                    break

                # Predict arguments for this tool
                predicted_args = self._predict_arguments(
                    tool_name,
                    signature,
                    result,
                )

                if not predicted_args:
                    continue

                # Create prefetch signature
                prefetch_sig = ToolCallSignature(
                    tool_name=tool_name,
                    arguments=predicted_args,
                    context_hash=signature.context_hash,
                )

                # Check if already cached
                cached_entry = await self._cache.get(prefetch_sig)
                if cached_entry is not None:
                    continue

                # Check if already pending
                task_key = self._make_task_key(prefetch_sig)
                if task_key in self._pending:
                    continue

                # Start prefetch task
                task = asyncio.create_task(self._prefetch(prefetch_sig))
                self._pending[task_key] = (task, time.time(), prefetch_sig)
                self._stats.prefetches_started += 1

                prefetch_results.append(
                    PrefetchResult(
                        tool_name=tool_name,
                        arguments=predicted_args,
                        success=True,
                        cached=False,
                        latency_ms=0.0,
                    )
                )

        return prefetch_results

    async def _prefetch(self, signature: ToolCallSignature) -> Any | None:
        """Execute a prefetch operation in background.

        Args:
            signature: Tool call to prefetch

        Returns:
            The result if successful, None on error
        """
        try:
            # Execute the tool
            result = await self._executor(
                signature.tool_name,
                signature.arguments,
            )

            # Estimate tokens (rough approximation)
            result_str = json.dumps(result, default=str)
            result_tokens = len(result_str) // 4

            # Cache the result
            await self._cache.put(
                signature,
                result,
                token_count=result_tokens,
            )

            self._stats.prefetches_completed += 1
            return result

        except Exception:
            self._stats.prefetches_failed += 1
            return None

        finally:
            # Remove from pending (done in cleanup, but belt-and-suspenders)
            task_key = self._make_task_key(signature)
            async with self._lock:
                self._pending.pop(task_key, None)

    def _predict_arguments(
        self,
        tool_name: str,
        prev_signature: ToolCallSignature,
        prev_result: Any,
    ) -> dict[str, Any] | None:
        """Predict likely arguments for a tool call.

        Uses a combination of result-based inference and learned
        parameter patterns to predict argument values.

        Args:
            tool_name: Tool to predict arguments for
            prev_signature: Previous tool call signature
            prev_result: Result from previous tool call

        Returns:
            Predicted arguments dict, or None if prediction fails
        """
        # Try result-based prediction first (more specific)
        result_args = self._predict_from_result(
            tool_name,
            prev_signature,
            prev_result,
        )
        if result_args:
            return result_args

        # Fall back to pattern-based prediction
        pattern_args = self._patterns.predict_arguments(tool_name)
        if pattern_args:
            return pattern_args

        return None

    def _predict_from_result(
        self,
        tool_name: str,
        prev_signature: ToolCallSignature,
        prev_result: Any,
    ) -> dict[str, Any] | None:
        """Predict arguments based on previous tool's result.

        Implements common tool chaining patterns:
        - search -> read_file: Use file path from search result
        - list_directory -> read_file: Use file from listing
        - read_file -> grep: Search within same file

        Args:
            tool_name: Target tool name
            prev_signature: Previous tool signature
            prev_result: Previous tool result

        Returns:
            Predicted arguments or None
        """
        prev_tool = prev_signature.tool_name.lower()
        target_tool = tool_name.lower()

        # search/grep/find -> read_file pattern
        search_to_read = target_tool in (
            "read_file",
            "read",
            "get_file",
        ) and prev_tool in ("search", "grep", "find", "glob", "list_directory")
        if search_to_read:
            path = self._extract_path_from_result(prev_result)
            if path:
                return {"path": path}

        # read_file -> grep pattern (search in same file)
        read_to_grep = target_tool in ("grep", "search") and prev_tool in (
            "read_file",
            "read",
        )
        if read_to_grep:
            path = prev_signature.arguments.get("path") or prev_signature.arguments.get(
                "file_path"
            )
            if path:
                # Get most common search pattern
                pattern_stats = self._patterns.get_parameter_pattern(
                    tool_name, "pattern"
                )
                pattern = pattern_stats.most_common_value() if pattern_stats else None
                if pattern:
                    return {"path": path, "pattern": pattern}

        # list_directory -> read_file (first file in listing)
        list_to_read = target_tool in ("read_file", "read") and prev_tool in (
            "list_directory",
            "ls",
            "list_dir",
        )
        if list_to_read:
            path = self._extract_path_from_result(prev_result)
            if path:
                return {"path": path}

        return None

    def _extract_path_from_result(self, result: Any) -> str | None:
        """Extract a file path from a tool result.

        Handles various result formats:
        - List of dicts with path/file keys
        - List of strings (file paths)
        - Dict with results/files list
        - Single dict with path

        Args:
            result: Tool result to extract path from

        Returns:
            Extracted path or None
        """
        # Handle list of results
        if isinstance(result, list) and result:
            first_item = result[0]

            # List of dicts with path field
            if isinstance(first_item, dict):
                for key in ("path", "file", "file_path", "filename", "name"):
                    if key in first_item:
                        return str(first_item[key])

            # List of strings (file paths)
            if isinstance(first_item, str):
                return first_item

        # Handle dict with results list
        if isinstance(result, dict):
            for key in ("results", "files", "matches", "items"):
                if key in result:
                    return self._extract_path_from_result(result[key])

            # Single result dict
            for key in ("path", "file", "file_path"):
                if key in result:
                    return str(result[key])

        return None

    async def await_prefetch(
        self,
        signature: ToolCallSignature,
        timeout_ms: float = 100.0,
    ) -> Any | None:
        """Wait for a matching prefetch to complete.

        Checks if there's a pending or completed prefetch that matches
        the requested tool call. If pending, waits up to timeout.

        Args:
            signature: Tool call signature to check
            timeout_ms: Maximum time to wait in milliseconds

        Returns:
            Cached result if available, None otherwise
        """
        task_key = self._make_task_key(signature)

        async with self._lock:
            pending_info = self._pending.get(task_key)

        if pending_info is not None:
            task, start_time, _ = pending_info
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=timeout_ms / 1000.0,
                )
                if result is not None:
                    self._stats.prefetch_hits += 1
                    # Calculate latency saved
                    elapsed_ms = (time.time() - start_time) * 1000
                    latency_saved = max(0, self._default_latency_ms - elapsed_ms)
                    self._stats.total_latency_saved_ms += latency_saved
                    return result
            except TimeoutError:
                pass

        # Check cache directly
        cached_entry = await self._cache.get(signature)
        if cached_entry is not None:
            self._stats.prefetch_hits += 1
            return cached_entry.result

        self._stats.prefetch_misses += 1
        return None

    async def cancel_pending(self) -> int:
        """Cancel all pending prefetch operations.

        Returns:
            Number of operations cancelled
        """
        async with self._lock:
            count = 0
            for _task_key, (task, _, _) in list(self._pending.items()):
                if not task.done():
                    task.cancel()
                    count += 1
            self._pending.clear()
            return count

    async def get_candidates(
        self,
        current_tool: str,
        top_k: int = 3,
    ) -> list[PrefetchCandidate]:
        """Get prefetch candidates without executing them.

        Useful for inspection and testing of prediction logic.

        Args:
            current_tool: The tool that was just called
            top_k: Number of candidates to return

        Returns:
            List of prefetch candidates with confidence scores
        """
        predictions = self._patterns.predict_next_tool(current_tool, top_k=top_k)

        candidates: list[PrefetchCandidate] = []
        for tool_name, confidence in predictions:
            predicted_args = self._patterns.predict_arguments(tool_name)
            candidates.append(
                PrefetchCandidate(
                    tool_name=tool_name,
                    predicted_arguments=predicted_args,
                    confidence=confidence,
                    expected_latency_ms=self._default_latency_ms,
                )
            )

        return candidates

    async def _cleanup_completed(self) -> None:
        """Remove completed tasks from pending dict."""
        completed_keys = [
            key for key, (task, _, _) in self._pending.items() if task.done()
        ]
        for key in completed_keys:
            self._pending.pop(key, None)

    def _make_task_key(self, signature: ToolCallSignature) -> str:
        """Create a unique key for a prefetch task.

        Args:
            signature: Tool call signature

        Returns:
            String key for task lookup
        """
        args_str = json.dumps(signature.arguments, sort_keys=True, default=str)
        return f"{signature.tool_name}:{hash(args_str)}"

    def clear_stats(self) -> None:
        """Reset prefetch statistics."""
        self._stats = PrefetchStats()
