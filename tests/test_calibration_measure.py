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
