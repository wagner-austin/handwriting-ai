from __future__ import annotations

from pathlib import Path

from PIL import Image
from pytest import MonkeyPatch

from handwriting_ai.training.calibration.calibrator import _DummyCfg
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.orchestrator import Orchestrator, OrchestratorConfig
from handwriting_ai.training.calibration.runner import BudgetConfig, CandidateOutcome
from handwriting_ai.training.dataset import PreprocessDataset


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, 0


class _FailingRunner:
    def __init__(self, fails: int) -> None:
        self._fails = int(fails)
        self.calls = 0

    def run(
        self,
        ds: PreprocessDataset,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        self.calls += 1
        if self._fails > 0:
            self._fails -= 1
            return CandidateOutcome(ok=False, res=None, error=None)
        res = CalibrationResult(
            intra_threads=cand.intra_threads,
            interop_threads=cand.interop_threads,
            num_workers=cand.num_workers,
            batch_size=cand.batch_size,
            samples_per_sec=1.0,
            p95_ms=1.0,
        )
        return CandidateOutcome(ok=True, res=res, error=None)


def test_orchestrator_preflight_and_abort(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    ds = PreprocessDataset(_FakeMNIST(8), _DummyCfg(batch_size=4))
    cands = [
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2),
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4),
    ]
    runner = _FailingRunner(fails=0)
    cfg = OrchestratorConfig(
        stage_a_budget=BudgetConfig(
            start_pct_max=10.0, abort_pct=95.0, timeout_s=1.0, max_failures=2
        ),
        stage_b_budget=BudgetConfig(
            start_pct_max=10.0, abort_pct=95.0, timeout_s=1.0, max_failures=2
        ),
        checkpoint_path=tmp_path / "ck.json",
    )
    orch = Orchestrator(runner, cfg)

    # Force preflight to fail to exercise abort path and checkpoint write
    def _pf_ok(_: float) -> bool:
        return False

    monkeypatch.setattr(orch, "_preflight_ok", _pf_ok, raising=False)
    res = orch.run_stage_a(ds, cands, samples=1)
    # Should have written a checkpoint after first failure, then aborted on second
    assert res == []
    assert (tmp_path / "ck.json").exists()


def test_orchestrator_resume_stage_a_and_b(tmp_path: Path) -> None:
    ds = PreprocessDataset(_FakeMNIST(8), _DummyCfg(batch_size=4))
    cands = [
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2),
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4),
    ]
    runner = _FailingRunner(fails=0)
    cfg = OrchestratorConfig(
        stage_a_budget=BudgetConfig(
            start_pct_max=99.0, abort_pct=95.0, timeout_s=1.0, max_failures=2
        ),
        stage_b_budget=BudgetConfig(
            start_pct_max=99.0, abort_pct=95.0, timeout_s=1.0, max_failures=2
        ),
        checkpoint_path=tmp_path / "ck.json",
    )
    orch = Orchestrator(runner, cfg)

    # Write stage A checkpoint to skip first candidate
    from handwriting_ai.training.calibration.checkpoint import (
        CalibrationCheckpoint,
        CalibrationStage,
        write_checkpoint,
    )

    prior = CalibrationResult(
        intra_threads=1,
        interop_threads=None,
        num_workers=0,
        batch_size=2,
        samples_per_sec=1.0,
        p95_ms=1.0,
    )
    write_checkpoint(
        cfg.checkpoint_path,
        CalibrationCheckpoint(
            stage=CalibrationStage.A, index=1, results=[prior], shortlist=None, seed=None
        ),
    )
    res_a = orch.run_stage_a(ds, cands, samples=1)
    # Prior plus one new result
    assert len(res_a) == 2

    # Now resume stage B with checkpoint
    write_checkpoint(
        cfg.checkpoint_path,
        CalibrationCheckpoint(
            stage=CalibrationStage.B, index=1, results=[prior], shortlist=None, seed=None
        ),
    )
    res_b = orch.run_stage_b(ds, res_a, samples=1)
    assert len(res_b) >= 1


def test_orchestrator_preflight_recovers_then_runs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # First preflight returns False, then True to exercise continue path
    ds = PreprocessDataset(_FakeMNIST(8), _DummyCfg(batch_size=4))
    cands = [
        Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2),
    ]

    class _Runner:
        def __init__(self) -> None:
            self.calls = 0

        def run(
            self,
            ds2: PreprocessDataset,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            self.calls += 1
            res = CalibrationResult(
                cand.intra_threads,
                cand.interop_threads,
                cand.num_workers,
                cand.batch_size,
                1.0,
                1.0,
            )
            return CandidateOutcome(ok=True, res=res, error=None)

    runner = _Runner()
    cfg = OrchestratorConfig(
        stage_a_budget=BudgetConfig(99.0, 95.0, 1.0, 2),
        stage_b_budget=BudgetConfig(99.0, 95.0, 1.0, 2),
        checkpoint_path=tmp_path / "ck.json",
    )
    orch = Orchestrator(runner, cfg)

    calls = {"i": 0}

    def _pf_ok(_: float) -> bool:
        calls["i"] += 1
        return calls["i"] > 1

    monkeypatch.setattr(orch, "_preflight_ok", _pf_ok, raising=False)
    out = orch.run_stage_a(ds, cands, samples=1)
    # One preflight failure (checkpoint) then success run
    assert len(out) == 1 and runner.calls == 1
    assert (tmp_path / "ck.json").exists()


def test_orchestrator_candidate_failure_aborts(tmp_path: Path) -> None:
    ds = PreprocessDataset(_FakeMNIST(4), _DummyCfg(batch_size=2))
    cands = [Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)]

    class _FailRunner:
        def run(
            self,
            ds2: PreprocessDataset,
            cand: Candidate,
            samples: int,
            budget: BudgetConfig,
        ) -> CandidateOutcome:
            from handwriting_ai.training.calibration.runner import CandidateError, CandidateOutcome

            return CandidateOutcome(
                ok=False,
                res=None,
                error=CandidateError(kind="timeout", message="candidate timed out", exit_code=None),
            )

    cfg = OrchestratorConfig(
        stage_a_budget=BudgetConfig(99.0, 95.0, 1.0, 1),
        stage_b_budget=BudgetConfig(99.0, 95.0, 1.0, 1),
        checkpoint_path=tmp_path / "ck.json",
    )
    orch = Orchestrator(_FailRunner(), cfg)
    res = orch.run_stage_a(ds, cands, samples=1)
    assert res == []
