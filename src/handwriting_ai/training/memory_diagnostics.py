from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

from handwriting_ai.logging import get_logger
from handwriting_ai.monitoring import MemorySnapshot, get_memory_snapshot


@dataclass(frozen=True)
class MemoryDiagnostics:
    """Diagnostic metrics for memory behavior over time."""

    current_mb: int
    peak_mb: int
    avg_mb: int
    growth_rate_mb_per_batch: float
    volatility_mb: float
    trend: Literal["stable", "growing", "critical"]
    batches_until_oom: int | None
    window_size: int


class MemoryHistory:
    """Tracks memory snapshots over time and computes diagnostics.

    Records a sliding window of (batch_index, usage_mb) points and computes
    growth rate, volatility, and OOM prediction based on the observed trend.
    """

    def __init__(self, window_size: int = 50) -> None:
        if window_size <= 0:
            window_size = 1
        self._window_size: int = int(window_size)
        self._snapshots: deque[tuple[int, int]] = deque(maxlen=self._window_size)
        self._batch_num: int = 0

    def record_batch(self, snapshot: MemorySnapshot | None = None) -> None:
        """Record memory snapshot for current batch.

        Accepts an optional pre-captured `MemorySnapshot` to avoid duplicate
        sampling by callers that already fetched it as part of their workflow.
        """
        snap = snapshot if snapshot is not None else get_memory_snapshot()
        usage_mb = int(snap.cgroup_usage.usage_bytes // (1024 * 1024))
        self._batch_num += 1
        self._snapshots.append((self._batch_num, usage_mb))

    def get_diagnostics(
        self, limit_mb: int, *, current_snapshot: MemorySnapshot | None = None
    ) -> MemoryDiagnostics:
        """Compute diagnostic metrics from recorded history.

        Requires the memory limit in MB to predict batches-to-OOM. Accepts
        an optional `current_snapshot` for current usage if history is empty.
        """
        if not self._snapshots:
            # History empty: compute a minimal diagnostics view based on current snapshot
            snap = current_snapshot if current_snapshot is not None else get_memory_snapshot()
            current_mb = int(snap.cgroup_usage.usage_bytes // (1024 * 1024))
            return MemoryDiagnostics(
                current_mb=current_mb,
                peak_mb=current_mb,
                avg_mb=current_mb,
                growth_rate_mb_per_batch=0.0,
                volatility_mb=0.0,
                trend="stable",
                batches_until_oom=_predict_oom(current_mb, limit_mb, 0.0),
                window_size=0,
            )

        memory_values = [mb for _, mb in self._snapshots]
        current_mb = int(memory_values[-1])
        peak_mb = int(max(memory_values))
        avg_mb = int(sum(memory_values) / len(memory_values))

        # Compute growth rate using linear regression
        growth_rate = _compute_growth_rate(self._snapshots)

        # Compute volatility (standard deviation of memory changes)
        volatility = _compute_volatility(memory_values)

        # Determine trend
        trend = _classify_trend(growth_rate, volatility)

        # Predict batches until OOM
        batches_until_oom = _predict_oom(current_mb, int(limit_mb), float(growth_rate))

        return MemoryDiagnostics(
            current_mb=current_mb,
            peak_mb=peak_mb,
            avg_mb=avg_mb,
            growth_rate_mb_per_batch=float(growth_rate),
            volatility_mb=float(volatility),
            trend=trend,
            batches_until_oom=batches_until_oom,
            window_size=len(self._snapshots),
        )

    def reset(self) -> None:
        """Clear history and reset batch counter."""
        self._snapshots.clear()
        self._batch_num = 0


def _compute_growth_rate(snapshots: deque[tuple[int, int]]) -> float:
    """Compute memory growth rate (MB/batch) using linear regression.

    Returns the slope of the best-fit line through (batch_num, memory_mb) points.
    """
    if len(snapshots) < 2:
        return 0.0

    n = float(len(snapshots))
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for batch_num, memory_mb in snapshots:
        x = float(batch_num)
        y = float(memory_mb)
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denominator = n * sum_x2 - (sum_x * sum_x)
    if denominator == 0.0:
        return 0.0

    slope = (n * sum_xy - (sum_x * sum_y)) / denominator
    return float(slope)


def _compute_volatility(memory_values: list[int]) -> float:
    """Compute memory volatility (standard deviation of batch-to-batch changes)."""
    if len(memory_values) < 2:
        return 0.0

    deltas = [int(memory_values[i] - memory_values[i - 1]) for i in range(1, len(memory_values))]

    mean_delta = float(sum(deltas)) / float(len(deltas))
    variance_f: float = sum((float(d) - mean_delta) ** 2 for d in deltas) / float(len(deltas))
    return float(math.sqrt(variance_f))


def _classify_trend(
    growth_rate: float, volatility: float
) -> Literal["stable", "growing", "critical"]:
    """Classify memory trend based on growth rate and volatility.

    - stable: growth < 1.0 MB/batch and volatility < 20 MB
    - growing: growth >= 1.0 MB/batch or volatility >= 20 MB
    - critical: growth >= 20.0 MB/batch or volatility >= 100 MB
    """
    if growth_rate >= 20.0 or volatility >= 100.0:
        return "critical"
    if growth_rate >= 1.0 or volatility >= 20.0:
        return "growing"
    return "stable"


def _predict_oom(current_mb: int, limit_mb: int, growth_rate: float) -> int | None:
    """Predict number of batches until OOM based on current growth rate.

    Returns None if growth rate is non-positive or prediction is > 1000 batches.
    """
    if growth_rate <= 0.0:
        return None

    remaining_mb = int(limit_mb - current_mb)
    if remaining_mb <= 0:
        return 0

    batches = float(remaining_mb) / float(growth_rate)
    return int(batches) if batches <= 1000.0 else None


_history: MemoryHistory | None = None


def _ensure_history() -> MemoryHistory:
    """Get or initialize the module-level memory history instance."""
    global _history
    if _history is None:
        _history = MemoryHistory()
    return _history


def initialize_diagnostics(window_size: int = 50) -> None:
    """Initialize memory diagnostics tracking with specified window size."""
    global _history
    _history = MemoryHistory(window_size)


def record_batch_memory(snapshot: MemorySnapshot | None = None) -> None:
    """Record memory snapshot for current batch. Auto-initializes if needed."""
    hist = _ensure_history()
    hist.record_batch(snapshot)


def get_memory_diagnostics(snapshot: MemorySnapshot | None = None) -> MemoryDiagnostics:
    """Get current memory diagnostics. Auto-initializes if needed."""
    hist = _ensure_history()
    snap = snapshot if snapshot is not None else get_memory_snapshot()
    limit_mb = int(snap.cgroup_usage.limit_bytes // (1024 * 1024))
    return hist.get_diagnostics(limit_mb, current_snapshot=snap)


def log_memory_diagnostics(*, context: str = "", snapshot: MemorySnapshot | None = None) -> None:
    """Log comprehensive memory diagnostics with optional context."""
    log = get_logger()
    diag = get_memory_diagnostics(snapshot)
    ctx = f"{context} " if context else ""

    log.info(
        f"{ctx}memory_diagnostics "
        f"current_mb={diag.current_mb} peak_mb={diag.peak_mb} avg_mb={diag.avg_mb} "
        f"growth_rate_mb_per_batch={diag.growth_rate_mb_per_batch:.3f} "
        f"volatility_mb={diag.volatility_mb:.1f} trend={diag.trend} "
        f"batches_until_oom={diag.batches_until_oom} window_size={diag.window_size}"
    )


def reset_diagnostics() -> None:
    """Reset memory diagnostics history."""
    global _history
    if _history is not None:
        _history.reset()


__all__ = [
    "MemoryDiagnostics",
    "MemoryHistory",
    "get_memory_diagnostics",
    "initialize_diagnostics",
    "log_memory_diagnostics",
    "record_batch_memory",
    "reset_diagnostics",
]
