from __future__ import annotations

from handwriting_ai.training.resources import (
    _compute_optimal_threads,
    _compute_optimal_workers,
    compute_max_batch_size,
)


def test_compute_max_batch_size_tiers() -> None:
    assert compute_max_batch_size(None) is None
    assert compute_max_batch_size(500 * 1024 * 1024) == 64
    assert compute_max_batch_size(1500 * 1024 * 1024) == 128
    assert compute_max_batch_size(3 * 1024 * 1024 * 1024) == 256
    assert compute_max_batch_size(8 * 1024 * 1024 * 1024) == 512


def test_optimal_threads_and_workers_small_cores() -> None:
    assert _compute_optimal_threads(1) == 1
    assert _compute_optimal_threads(2) == 2
    assert _compute_optimal_workers(1, None) == 0
    assert _compute_optimal_workers(2, None) == 0


def test_optimal_threads_and_workers_medium_cores() -> None:
    assert _compute_optimal_threads(4) in (3, 4)
    assert _compute_optimal_threads(8) <= 8
    assert _compute_optimal_workers(4, None) <= 2
    assert _compute_optimal_workers(8, None) <= 2
