from __future__ import annotations

from handwriting_ai.training.calibrate import _candidate_workers
from handwriting_ai.training.resources import ResourceLimits


def test_candidate_workers_low_memory_returns_only_zero() -> None:
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1 * 1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    ws = _candidate_workers(limits)
    assert ws == [0]
