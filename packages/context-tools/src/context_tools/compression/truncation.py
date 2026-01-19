"""Smart list truncation with representative sampling.

Provides intelligent truncation strategies that preserve important
information while reducing token usage for large lists.

Example:
    >>> truncator = ListTruncator()
    >>> data = list(range(100))
    >>> result = truncator.truncate(data, keep=10, strategy=TruncationStrategy.UNIFORM)
    >>> print(result.items)  # [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

from context_tools.types import (
    StatisticalSummary,
    TruncationResult,
    TruncationStrategy,
    TypeDistribution,
)


class ListTruncator:
    """Smart list truncation with multiple strategies.

    Provides intelligent truncation that preserves representative
    samples and important statistical information.

    Attributes:
        default_strategy: Default truncation strategy
        include_statistics: Whether to compute statistics for numeric lists
        include_type_distribution: Whether to track type distribution
        seed: Random seed for reproducible sampling
    """

    def __init__(
        self,
        default_strategy: TruncationStrategy = TruncationStrategy.HEAD_TAIL,
        include_statistics: bool = True,
        include_type_distribution: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize list truncator.

        Args:
            default_strategy: Default strategy for truncation
            include_statistics: Compute stats for numeric lists
            include_type_distribution: Track type breakdown
            seed: Random seed for reproducible results
        """
        self._default_strategy = default_strategy
        self._include_statistics = include_statistics
        self._include_type_distribution = include_type_distribution
        self._seed = seed
        self._rng = random.Random(seed)

    def truncate(
        self,
        data: list[Any],
        keep: int,
        strategy: TruncationStrategy | None = None,
    ) -> TruncationResult:
        """Truncate a list using the specified strategy.

        Args:
            data: List to truncate
            keep: Number of items to keep
            strategy: Truncation strategy (uses default if None)

        Returns:
            TruncationResult with kept items and metadata
        """
        if not data:
            return TruncationResult(
                items=[],
                original_count=0,
                kept_count=0,
                omitted_count=0,
                strategy=strategy or self._default_strategy,
                is_truncated=False,
            )

        original_count = len(data)

        # No truncation needed
        if original_count <= keep:
            return TruncationResult(
                items=list(data),
                original_count=original_count,
                kept_count=original_count,
                omitted_count=0,
                strategy=strategy or self._default_strategy,
                sample_indices=list(range(original_count)),
                is_truncated=False,
            )

        strategy = strategy or self._default_strategy

        # Select items based on strategy
        if strategy == TruncationStrategy.HEAD_TAIL:
            items, indices = self._head_tail_sample(data, keep)
        elif strategy == TruncationStrategy.UNIFORM:
            items, indices = self._uniform_sample(data, keep)
        elif strategy == TruncationStrategy.RESERVOIR:
            items, indices = self._reservoir_sample(data, keep)
        elif strategy == TruncationStrategy.DIVERSE:
            items, indices = self._diverse_sample(data, keep)
        elif strategy == TruncationStrategy.STRATIFIED:
            items, indices = self._stratified_sample(data, keep)
        else:
            items, indices = self._head_tail_sample(data, keep)

        # Compute statistics if applicable
        stats = None
        if self._include_statistics and self._is_numeric_list(data):
            stats = self._compute_statistics(data)

        # Compute type distribution if applicable
        type_dist = None
        if self._include_type_distribution:
            type_dist = self._compute_type_distribution(data)

        return TruncationResult(
            items=items,
            original_count=original_count,
            kept_count=len(items),
            omitted_count=original_count - len(items),
            strategy=strategy,
            statistical_summary=stats,
            type_distribution=type_dist,
            sample_indices=indices,
            is_truncated=True,
        )

    def _head_tail_sample(
        self,
        data: list[Any],
        keep: int,
    ) -> tuple[list[Any], list[int]]:
        """Keep first and last items.

        Classic truncation that preserves temporal boundaries.
        """
        half = keep // 2
        remainder = keep - 2 * half

        head_count = half + remainder
        tail_count = half

        head_indices = list(range(head_count))
        tail_indices = list(range(len(data) - tail_count, len(data)))

        indices = head_indices + tail_indices
        items = [data[i] for i in indices]

        return items, indices

    def _uniform_sample(
        self,
        data: list[Any],
        keep: int,
    ) -> tuple[list[Any], list[int]]:
        """Evenly spaced samples across the list.

        Provides uniform coverage of the entire list.
        """
        n = len(data)
        if keep >= n:
            return list(data), list(range(n))

        # Calculate step size
        step = (n - 1) / (keep - 1) if keep > 1 else 0

        indices = []
        for i in range(keep):
            idx = min(int(i * step), n - 1)
            if idx not in indices:
                indices.append(idx)
            else:
                # Find next available index
                for j in range(idx + 1, n):
                    if j not in indices:
                        indices.append(j)
                        break

        # Ensure we have enough indices
        while len(indices) < keep and len(indices) < n:
            for j in range(n):
                if j not in indices:
                    indices.append(j)
                    break

        indices.sort()
        items = [data[i] for i in indices]

        return items, indices

    def _reservoir_sample(
        self,
        data: list[Any],
        keep: int,
    ) -> tuple[list[Any], list[int]]:
        """Random reservoir sampling (Algorithm R).

        Each item has equal probability of being selected.
        """
        n = len(data)
        if keep >= n:
            return list(data), list(range(n))

        # Use deterministic seed based on data for reproducibility
        if self._seed is None:
            seed = self._hash_data(data)
        else:
            # Combine user seed with data hash for deterministic results per data
            seed = self._seed ^ self._hash_data(data)
        rng = random.Random(seed)

        # Reservoir sampling
        indices = list(range(keep))

        for i in range(keep, n):
            j = rng.randint(0, i)
            if j < keep:
                indices[j] = i

        indices.sort()
        items = [data[i] for i in indices]

        return items, indices

    def _diverse_sample(
        self,
        data: list[Any],
        keep: int,
    ) -> tuple[list[Any], list[int]]:
        """Maximize diversity in selection.

        For objects: select items with different key/value combinations.
        For primitives: select items with diverse values.
        """
        n = len(data)
        if keep >= n:
            return list(data), list(range(n))

        # Group items by their "signature"
        signatures: dict[str, list[int]] = {}
        for i, item in enumerate(data):
            sig = self._item_signature(item)
            if sig not in signatures:
                signatures[sig] = []
            signatures[sig].append(i)

        # Select from each signature group
        indices: list[int] = []
        sig_keys = list(signatures.keys())

        # Round-robin selection from signature groups
        round_idx = 0
        while len(indices) < keep:
            sig = sig_keys[round_idx % len(sig_keys)]
            group = signatures[sig]

            # Find next unselected from this group
            for idx in group:
                if idx not in indices:
                    indices.append(idx)
                    break

            round_idx += 1

            # Safety: if we've gone through all groups without adding
            if round_idx > len(sig_keys) * (len(data) // len(sig_keys) + 1):
                # Fall back to uniform for remaining
                for i in range(n):
                    if i not in indices:
                        indices.append(i)
                        if len(indices) >= keep:
                            break
                break

        indices = indices[:keep]
        indices.sort()
        items = [data[i] for i in indices]

        return items, indices

    def _stratified_sample(
        self,
        data: list[Any],
        keep: int,
    ) -> tuple[list[Any], list[int]]:
        """Stratified sampling by type.

        Maintains proportional representation of each type.
        """
        n = len(data)
        if keep >= n:
            return list(data), list(range(n))

        # Group by type
        type_groups: dict[str, list[int]] = {}
        for i, item in enumerate(data):
            t = self._item_type(item)
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(i)

        # Allocate proportionally
        indices: list[int] = []
        allocations: dict[str, int] = {}

        for t, group in type_groups.items():
            # Proportional allocation
            alloc = max(1, int(keep * len(group) / n))
            allocations[t] = min(alloc, len(group))

        # Adjust to hit target
        total_alloc = sum(allocations.values())
        if total_alloc < keep:
            # Add more from largest groups
            diff = keep - total_alloc
            sorted_types = sorted(
                type_groups.keys(), key=lambda t: len(type_groups[t]), reverse=True
            )
            for t in sorted_types:
                if diff <= 0:
                    break
                available = len(type_groups[t]) - allocations[t]
                add = min(diff, available)
                allocations[t] += add
                diff -= add

        # Select from each group
        for t, group in type_groups.items():
            alloc = allocations.get(t, 0)
            if alloc > 0:
                # Uniform sample from group
                step = max(1, len(group) // alloc)
                for i in range(alloc):
                    idx = min(i * step, len(group) - 1)
                    indices.append(group[idx])

        indices = indices[:keep]
        indices.sort()
        items = [data[i] for i in indices]

        return items, indices

    def _item_signature(self, item: Any) -> str:
        """Generate a signature for diversity grouping."""
        if isinstance(item, dict):
            # Signature based on keys present
            keys = tuple(sorted(item.keys()))
            return f"dict:{hash(keys)}"
        elif isinstance(item, (list, tuple)):
            return f"list:{len(item)}"
        elif isinstance(item, bool):
            return f"bool:{item}"
        elif isinstance(item, (int, float)):
            # Bucket numeric values
            if item == 0:
                return "num:zero"
            elif item < 0:
                return f"num:neg:{int(math.log10(abs(item)) + 1) if item != 0 else 0}"
            else:
                return f"num:pos:{int(math.log10(item) + 1) if item > 0 else 0}"
        elif isinstance(item, str):
            # Bucket by length
            return f"str:{len(item) // 10 * 10}"
        elif item is None:
            return "null"
        else:
            return f"other:{type(item).__name__}"

    def _item_type(self, item: Any) -> str:
        """Get type name for stratification."""
        if item is None:
            return "null"
        elif isinstance(item, bool):
            return "boolean"
        elif isinstance(item, int):
            return "integer"
        elif isinstance(item, float):
            return "number"
        elif isinstance(item, str):
            return "string"
        elif isinstance(item, dict):
            return "object"
        elif isinstance(item, (list, tuple)):
            return "array"
        else:
            return "other"

    def _is_numeric_list(self, data: list[Any]) -> bool:
        """Check if list contains only numeric values."""
        if not data:
            return False
        return all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in data
        )

    def _compute_statistics(self, data: list[Any]) -> StatisticalSummary:
        """Compute statistics for numeric list."""
        # Filter to numeric only
        nums = [
            x for x in data if isinstance(x, (int, float)) and not isinstance(x, bool)
        ]

        if not nums:
            return StatisticalSummary(
                count=0,
                min_value=0.0,
                max_value=0.0,
                mean=0.0,
            )

        n = len(nums)
        sorted_nums = sorted(nums)

        min_val = float(sorted_nums[0])
        max_val = float(sorted_nums[-1])
        total = sum(nums)
        mean = total / n

        # Median
        if n % 2 == 1:
            median = float(sorted_nums[n // 2])
        else:
            median = float((sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2)

        # Standard deviation
        std_dev = None
        if n > 1:
            variance = sum((x - mean) ** 2 for x in nums) / (n - 1)
            std_dev = math.sqrt(variance)

        # Percentiles (25th and 75th)
        percentiles = {}
        for p in [25, 75]:
            idx = (n - 1) * p / 100
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            percentiles[p] = float(
                sorted_nums[lower] * (1 - weight) + sorted_nums[upper] * weight
            )

        return StatisticalSummary(
            count=n,
            min_value=min_val,
            max_value=max_val,
            mean=mean,
            std_dev=std_dev,
            median=median,
            sum_value=total,
            percentiles=percentiles,
        )

    def _compute_type_distribution(self, data: list[Any]) -> TypeDistribution:
        """Compute type distribution of list items."""
        type_counts: dict[str, int] = {}

        for item in data:
            t = self._item_type(item)
            type_counts[t] = type_counts.get(t, 0) + 1

        return TypeDistribution(
            type_counts=type_counts,
            total_items=len(data),
        )

    def _hash_data(self, data: list[Any]) -> int:
        """Generate deterministic hash for reproducible sampling."""
        try:
            data_str = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            data_str = str(data)
        return int(hashlib.md5(data_str.encode()).hexdigest()[:8], 16)

    def auto_select_strategy(self, data: list[Any]) -> TruncationStrategy:
        """Automatically select best strategy for the data.

        Args:
            data: List to analyze

        Returns:
            Recommended truncation strategy
        """
        if not data:
            return TruncationStrategy.HEAD_TAIL

        # Check if numeric
        if self._is_numeric_list(data):
            return TruncationStrategy.UNIFORM

        # Check type heterogeneity
        type_dist = self._compute_type_distribution(data)
        if not type_dist.is_homogeneous:
            return TruncationStrategy.STRATIFIED

        # Check if objects with varying structures
        if all(isinstance(x, dict) for x in data):
            signatures = {self._item_signature(x) for x in data}
            if len(signatures) > 1:
                return TruncationStrategy.DIVERSE

        # Default for homogeneous lists
        return TruncationStrategy.UNIFORM
