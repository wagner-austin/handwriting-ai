from __future__ import annotations

from PIL import Image

from handwriting_ai.training.calibration import measure as _measure
from handwriting_ai.training.calibration.calibrator import _DummyCfg
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.safety import (
    MemoryGuardConfig,
    reset_memory_guard,
    set_memory_guard_config,
)


class _FakeMNIST:
    def __init__(self, n: int = 64) -> None:
        self._n = n

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        # 28x28 grayscale image, label 0
        img = Image.new("L", (28, 28), color=0)
        return img, 0


def test_measure_candidate_basic_runs() -> None:
    base = _FakeMNIST(64)
    ds = PreprocessDataset(base, _DummyCfg(batch_size=32))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=32)
    res = _measure._measure_candidate(ds, cand, samples=2)
    assert res.batch_size >= 1
    assert res.samples_per_sec >= 0.0
    assert res.p95_ms >= 0.0


def test_measure_training_zero_length_loader() -> None:
    # Call internal helper with an empty iterator to exercise the early-return path
    from collections.abc import Generator

    import torch as _t

    def _empty_loader() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
        if False:  # pragma: no cover - never executed
            yield _t.zeros((1, 1, 28, 28)), _t.zeros((1,), dtype=_t.long)

    sps_f: float
    p95_f: float
    peak_f: float
    exceeded_b: bool
    sps_f, p95_f, peak_f, exceeded_b = _measure._measure_training(
        ds_len=0,
        loader=_empty_loader(),
        k=2,
        device=_t.device("cpu"),
        batch_size_hint=16,
    )
    assert sps_f == 0.0 and p95_f == 0.0 and peak_f == 0.0 and exceeded_b is False


def test_measure_candidate_exceeds_threshold_backoff() -> None:
    # Force threshold to 0 so any usage triggers backoff path in binary search
    set_memory_guard_config(
        MemoryGuardConfig(enabled=True, threshold_percent=0.0, required_consecutive=1)
    )
    reset_memory_guard()
    try:
        base = _FakeMNIST(8)
        ds = PreprocessDataset(base, _DummyCfg(batch_size=8))
        cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=8)
        res = _measure._measure_candidate(ds, cand, samples=1)
        # Expect the algorithm to back off from the initial size due to threshold=0.0
        assert res.batch_size < 8
    finally:
        # Restore default guard disabled to avoid cross-test effects
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=3)
        )
        reset_memory_guard()


def test_measure_loader_break_on_exhaustion() -> None:
    # Loader yields fewer batches than requested n_batches, triggers inner break path
    from collections.abc import Generator

    import torch as _t

    from handwriting_ai.training import calibrate as cal

    def _loader_once() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
        x = _t.zeros((1, 1, 28, 28), dtype=_t.float32)
        y = _t.zeros((1,), dtype=_t.long)
        yield x, y

    sps, p95 = cal._measure_loader(100, _loader_once(), 3, batch_size_hint=2)
    assert sps >= 0.0 and p95 >= 0.0


def test_measure_candidate_multiple_batch_sizes_no_leak() -> None:
    """Stress test: ensure memory doesn't accumulate across multiple batch size tests.

    This simulates the binary search behavior where multiple batch sizes are tested
    sequentially. Without proper DataLoader cleanup, memory would accumulate.
    """
    from handwriting_ai.monitoring import get_memory_snapshot

    base = _FakeMNIST(256)
    ds = PreprocessDataset(base, _DummyCfg(batch_size=64))

    # Measure initial memory baseline (unused in range-based check; retained for context)
    _ = get_memory_snapshot()

    # Test multiple batch sizes sequentially (simulating binary search)
    batch_sizes = [64, 32, 16, 8, 4, 2]
    memory_readings: list[int] = []

    for bs in batch_sizes:
        cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=bs)
        _measure._measure_candidate(ds, cand, samples=1)

        # Measure memory after each test
        snap = get_memory_snapshot()
        current_mb = snap.main_process.rss_bytes // (1024 * 1024)
        memory_readings.append(current_mb)

    # Verify memory didn't accumulate significantly across the sequence itself
    # Allow some variance (50MB) for legitimate allocations, but not unbounded growth
    seq_max = max(memory_readings)
    seq_min = min(memory_readings)
    seq_growth_mb = seq_max - seq_min

    # If a leak exists across repeated calibrations, the sequence range would exceed 50MB
    assert seq_growth_mb < 50, (
        f"Memory accumulation detected across sequence: grew {seq_growth_mb}MB "
        f"within readings {memory_readings}"
    )

    # Also verify memory didn't monotonically increase (sign of leak)
    # With proper cleanup, later tests shouldn't use more memory than earlier ones
    first_half_avg = sum(memory_readings[:3]) / 3
    second_half_avg = sum(memory_readings[3:]) / 3
    avg_increase = second_half_avg - first_half_avg

    assert avg_increase < 50, (
        f"Memory accumulation detected: second half avg ({second_half_avg:.1f}MB) "
        f"significantly higher than first half avg ({first_half_avg:.1f}MB). "
        f"Increase: {avg_increase:.1f}MB. Readings: {memory_readings}"
    )


def test_measure_candidate_with_workers_no_leak() -> None:
    """Stress test: ensure DataLoader workers are properly cleaned up between tests.

    Production calibration uses num_workers > 0 in containers with sufficient memory.
    This test verifies that worker processes are terminated and their memory is released
    when DataLoader is deleted between batch size attempts.
    """
    import time

    from handwriting_ai.monitoring import get_memory_snapshot

    base = _FakeMNIST(256)
    ds = PreprocessDataset(base, _DummyCfg(batch_size=64))

    # Test multiple batch sizes with worker processes (production-like)
    batch_sizes = [64, 32, 16, 8, 4, 2]
    memory_readings: list[int] = []

    for bs in batch_sizes:
        cand = Candidate(intra_threads=1, interop_threads=None, num_workers=1, batch_size=bs)
        _measure._measure_candidate(ds, cand, samples=1)

        # Allow worker processes time to fully terminate after DataLoader cleanup
        # Worker shutdown is asynchronous - wait for OS to reclaim process resources
        time.sleep(0.15)

        # Measure memory after worker cleanup completes
        snap = get_memory_snapshot()
        current_mb = snap.main_process.rss_bytes // (1024 * 1024)
        memory_readings.append(current_mb)

    # Verify memory didn't accumulate across the sequence
    # Workers add overhead, so allow more variance (75MB) than the no-worker test
    seq_max = max(memory_readings)
    seq_min = min(memory_readings)
    seq_growth_mb = seq_max - seq_min

    # If workers aren't cleaned up, memory would grow 150+ MB across 6 tests
    # With proper cleanup, sequence range should remain bounded even with worker overhead
    assert seq_growth_mb < 75, (
        f"Worker memory leak detected: sequence grew {seq_growth_mb}MB "
        f"across {len(batch_sizes)} batch size tests with num_workers=1. "
        f"Readings: {memory_readings}"
    )

    # Verify no monotonic memory increase indicating worker accumulation
    first_half_avg = sum(memory_readings[:3]) / 3
    second_half_avg = sum(memory_readings[3:]) / 3
    avg_increase = second_half_avg - first_half_avg

    assert avg_increase < 75, (
        f"Worker accumulation detected: second half avg ({second_half_avg:.1f}MB) "
        f"significantly higher than first half avg ({first_half_avg:.1f}MB). "
        f"Increase: {avg_increase:.1f}MB. Readings: {memory_readings}"
    )
