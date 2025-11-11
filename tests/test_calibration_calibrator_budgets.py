from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai.training.calibration.calibrator import calibrate_input_pipeline as _cal
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import OrchestratorConfig
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.resources import ResourceLimits


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


class _DummyRunner:
    def __init__(self) -> None:
        self.config: OrchestratorConfig | None = None

    def __call__(self, *args: object, **kwargs: object) -> None:  # pragma: no cover - not used
        raise AssertionError


def test_calibrator_low_mem_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch orchestrator to capture budgets and avoid subprocess work
    captured: list[OrchestratorConfig] = []

    class _Orch:
        def __init__(self, *, runner: object, config: OrchestratorConfig) -> None:
            captured.append(config)

        def run_stage_a(
            self, ds: PreprocessDataset, cands: list[Candidate], samples: int
        ) -> list[CalibrationResult]:
            # return one result per candidate
            return [
                CalibrationResult(
                    intra_threads=int(c.intra_threads),
                    interop_threads=c.interop_threads,
                    num_workers=int(c.num_workers),
                    batch_size=int(c.batch_size),
                    samples_per_sec=1.0,
                    p95_ms=1.0,
                )
                for c in cands
            ]

        def run_stage_b(
            self, ds: PreprocessDataset, shortlist: list[CalibrationResult], samples: int
        ) -> list[CalibrationResult]:
            return shortlist

    class _Snap:
        class _Usage:
            limit_bytes = 1024 * 1024 * 1024  # 1GB

        cgroup_usage = _Usage()

    monkeypatch.setattr("handwriting_ai.training.calibration.calibrator.Orchestrator", _Orch)
    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.get_memory_snapshot", lambda: _Snap()
    )
    base = _FakeMNIST(8)
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    _cal(
        train_base=base,
        limits=limits,
        requested_batch_size=4,
        samples=1,
        cache_path=Path("/tmp/calib.json"),
        ttl_seconds=0,
        force=True,
    )
    cfg = captured[0]
    assert cfg.stage_a_budget.start_pct_max == 80.0
    assert cfg.stage_b_budget.abort_pct == 88.0


def test_calibrator_high_mem_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[OrchestratorConfig] = []

    class _Orch:
        def __init__(self, *, runner: object, config: OrchestratorConfig) -> None:
            captured.append(config)

        def run_stage_a(
            self, ds: PreprocessDataset, cands: list[Candidate], samples: int
        ) -> list[CalibrationResult]:
            out: list[CalibrationResult] = []
            for c in cands:
                out.append(
                    CalibrationResult(
                        int(c.intra_threads),
                        c.interop_threads,
                        int(c.num_workers),
                        int(c.batch_size),
                        1.0,
                        1.0,
                    )
                )
            return out

        def run_stage_b(
            self, ds: PreprocessDataset, shortlist: list[CalibrationResult], samples: int
        ) -> list[CalibrationResult]:
            return shortlist

    class _Snap:
        class _Usage:
            limit_bytes = 4 * 1024 * 1024 * 1024  # 4GB

        cgroup_usage = _Usage()

    monkeypatch.setattr("handwriting_ai.training.calibration.calibrator.Orchestrator", _Orch)
    monkeypatch.setattr(
        "handwriting_ai.training.calibration.calibrator.get_memory_snapshot", lambda: _Snap()
    )
    base = _FakeMNIST(8)
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=4 * 1024 * 1024 * 1024,
        optimal_threads=2,
        optimal_workers=1,
        max_batch_size=None,
    )
    _cal(
        train_base=base,
        limits=limits,
        requested_batch_size=4,
        samples=1,
        cache_path=Path("/tmp/calib.json"),
        ttl_seconds=0,
        force=True,
    )
    cfg = captured[0]
    assert cfg.stage_a_budget.start_pct_max == 85.0
    assert cfg.stage_b_budget.abort_pct == 92.0
