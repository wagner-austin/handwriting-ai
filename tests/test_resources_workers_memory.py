from __future__ import annotations

import handwriting_ai.training.resources as res


def test_compute_optimal_workers_memory_tiers() -> None:
    # < 2 GB -> 0 workers
    assert res._compute_optimal_workers(cores=2, memory_bytes=1024 * 1024 * 1024) == 0
    # 2-4 GB -> 1 worker (if cores >= 2)
    assert res._compute_optimal_workers(cores=4, memory_bytes=3 * 1024 * 1024 * 1024) == 1
    # >= 4 GB -> up to 2 workers, bounded by cores//2
    assert res._compute_optimal_workers(cores=4, memory_bytes=8 * 1024 * 1024 * 1024) == 2
    assert res._compute_optimal_workers(cores=3, memory_bytes=8 * 1024 * 1024 * 1024) == 1
