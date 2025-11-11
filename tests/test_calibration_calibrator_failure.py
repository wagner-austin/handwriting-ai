from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai.training.calibration.calibrator import (
    CalibrationError,
)
from handwriting_ai.training.calibration.calibrator import (
    calibrate_input_pipeline as _cal,
)
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.resources import ResourceLimits


class _FakeMNIST:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        from PIL import Image

        return Image.new("L", (28, 28), 0), 0


def test_calibrator_raises_on_empty_stage_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Orch:
        def __init__(self, *, runner, config):
            pass

        def run_stage_a(self, ds, cands, samples):
            return []

        def run_stage_b(self, ds, shortlist, samples):
            return []

    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.Orchestrator", _Orch
    )

    base = _FakeMNIST(4)
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    with pytest.raises(CalibrationError):
        _cal(
            train_base=base,
            limits=limits,
            requested_batch_size=4,
            samples=1,
            cache_path=tmp_path / "calibration.json",
            ttl_seconds=0,
            force=True,
        )


def test_calibrator_raises_on_empty_stage_b(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Orch:
        def __init__(self, *, runner, config):
            pass

        def run_stage_a(self, ds, cands, samples):
            # Return one measured result so Stage B runs
            return [
                CalibrationResult(1, None, 0, 2, 1.0, 1.0),
            ]

        def run_stage_b(self, ds, shortlist, samples):
            return []

    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.Orchestrator", _Orch
    )

    base = _FakeMNIST(4)
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    with pytest.raises(CalibrationError):
        _cal(
            train_base=base,
            limits=limits,
            requested_batch_size=4,
            samples=1,
            cache_path=tmp_path / "calibration.json",
            ttl_seconds=0,
            force=True,
        )


