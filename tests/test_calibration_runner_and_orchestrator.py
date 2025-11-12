from __future__ import annotations

import time
from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai.training.calibration.calibrator import _DummyCfg
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
)
from handwriting_ai.training.calibration.runner import (
    BudgetConfig,
    CandidateOutcome,
    SubprocessRunner,
)
from handwriting_ai.training.dataset import PreprocessDataset


class _FakeMNIST:
    def __init__(self, n: int = 32, *, sleep_s: float = 0.0, fail: bool = False) -> None:
        self._n = n
        self._sleep = float(sleep_s)
        self._fail = bool(fail)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        if self._fail:
            raise RuntimeError("fail-item")
        if self._sleep > 0:
            time.sleep(self._sleep)
        img = Image.new("L", (28, 28), color=0)
        return img, int(idx % 10)


def _mk_ds(n: int = 32, *, sleep_s: float = 0.0, fail: bool = False) -> PreprocessDataset:
    base = _FakeMNIST(n=n, sleep_s=sleep_s, fail=fail)
    return PreprocessDataset(base, _DummyCfg(batch_size=8))


def test_subprocess_runner_success() -> None:
    ds = _mk_ds(32)
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=20.0, max_failures=2)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=8)
    out = runner.run(ds, cand, samples=1, budget=budget)
    assert out.ok and out.res is not None
    assert isinstance(out.res, CalibrationResult)


def test_subprocess_runner_timeout() -> None:
    # Build inline spec with per-item sleep so child exceeds timeout
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.25, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=0.2, max_failures=1)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)
    out = runner.run(spec, cand, samples=1, budget=budget)
    assert not out.ok and out.error is not None
    assert out.error.kind == "timeout"


def test_subprocess_runner_runtime_error() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.0, fail=True),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    runner = SubprocessRunner()
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=95.0, timeout_s=10.0, max_failures=1)
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)
    out = runner.run(spec, cand, samples=1, budget=budget)
    assert not out.ok and out.error is not None
    assert out.error.kind in {"runtime", "oom", "timeout"}


def test_orchestrator_stage_flow_and_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dummy runner that fails once then succeeds
    class _DummyRunner:
        def __init__(self) -> None:
            self._calls = 0

        def run(
            self,
            ds: PreprocessDataset | PreprocessSpec,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            self._calls += 1
            if self._calls == 1:
                return CandidateOutcome(ok=False, res=None, error=None)
            res = CalibrationResult(
                intra_threads=cand.intra_threads,
                interop_threads=cand.interop_threads,
                num_workers=cand.num_workers,
                batch_size=cand.batch_size,
                samples_per_sec=10.0,
                p95_ms=5.0,
            )
            return CandidateOutcome(ok=True, res=res, error=None)

    ds = _mk_ds(16)
    cands = [
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4),
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=8),
    ]
    cfg = OrchestratorConfig(
        stage_a_budget=BudgetConfig(
            start_pct_max=99.0, abort_pct=95.0, timeout_s=2.0, max_failures=2
        ),
        stage_b_budget=BudgetConfig(
            start_pct_max=99.0, abort_pct=95.0, timeout_s=2.0, max_failures=2
        ),
        checkpoint_path=tmp_path / "calib.ckpt.json",
    )
    orch = Orchestrator(_DummyRunner(), cfg)
    res_a = orch.run_stage_a(ds, cands, samples=1)
    assert len(res_a) >= 1
    best = Orchestrator.select_best(res_a)
    assert best.batch_size in {4, 8}


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from handwriting_ai.training.calibration.checkpoint import (
        CalibrationCheckpoint,
        CalibrationStage,
        read_checkpoint,
        write_checkpoint,
    )

    res = CalibrationResult(
        intra_threads=1,
        interop_threads=None,
        num_workers=0,
        batch_size=4,
        samples_per_sec=3.14,
        p95_ms=7.5,
    )
    ck = CalibrationCheckpoint(
        stage=CalibrationStage.A,
        index=1,
        results=[res],
        shortlist=None,
        seed=None,
    )
    path = tmp_path / "ck.json"
    write_checkpoint(path, ck)
    ck2 = read_checkpoint(path)
    assert ck2 is not None and ck2.stage == CalibrationStage.A
    assert len(ck2.results) == 1 and ck2.results[0].batch_size == 4
