from __future__ import annotations

import gc as _gc
import time as _time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from handwriting_ai.logging import get_logger
from handwriting_ai.monitoring import get_memory_snapshot
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.checkpoint import (
    CalibrationCheckpoint,
    CalibrationStage,
    read_checkpoint,
    write_checkpoint,
)
from handwriting_ai.training.calibration.ds_spec import PreprocessSpec
from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.runner import (
    BudgetConfig,
    CandidateOutcome,
    CandidateRunner,
)
from handwriting_ai.training.dataset import PreprocessDataset


@dataclass(frozen=True)
class OrchestratorConfig:
    stage_a_budget: BudgetConfig
    stage_b_budget: BudgetConfig
    checkpoint_path: Path


class Orchestrator:
    def __init__(self, runner: CandidateRunner, config: OrchestratorConfig) -> None:
        self._runner = runner
        self._cfg = config
        self._log = get_logger()

    def _preflight_ok(self, start_pct_max: float) -> bool:
        snap = get_memory_snapshot()
        return float(snap.cgroup_usage.percent) <= float(start_pct_max)

    def _gc_and_log(self, stage: str, idx: int) -> None:
        _gc.collect()
        snap = get_memory_snapshot()
        mem_pct = float(snap.cgroup_usage.percent)
        main_mb = snap.main_process.rss_bytes // (1024 * 1024)
        workers_mb = sum(w.rss_bytes for w in snap.workers) // (1024 * 1024)
        cgroup_mb = snap.cgroup_usage.usage_bytes // (1024 * 1024)
        anon_mb = snap.cgroup_breakdown.anon_bytes // (1024 * 1024)
        file_mb = snap.cgroup_breakdown.file_bytes // (1024 * 1024)
        kernel_mb = snap.cgroup_breakdown.kernel_bytes // (1024 * 1024)
        slab_mb = snap.cgroup_breakdown.slab_bytes // (1024 * 1024)
        self._log.info(
            (
                "calibration_between_candidates_gc stage=%s idx=%d mem_pct=%.1f "
                "cgroup_mb=%d main_rss_mb=%d workers_rss_mb=%d "
                "anon_mb=%d file_mb=%d kernel_mb=%d slab_mb=%d"
            ),
            stage,
            idx,
            mem_pct,
            cgroup_mb,
            main_mb,
            workers_mb,
            anon_mb,
            file_mb,
            kernel_mb,
            slab_mb,
        )

    def _run_stage(
        self,
        stage: CalibrationStage,
        ds: PreprocessDataset | PreprocessSpec,
        items: Iterable[Candidate],
        samples: int,
        budget: BudgetConfig,
        resume_from: int = 0,
        prior_results: list[CalibrationResult] | None = None,
    ) -> list[CalibrationResult]:
        results: list[CalibrationResult] = [] if prior_results is None else list(prior_results)
        failures = 0
        for i, cand in enumerate(items):
            if i < resume_from:
                continue

            # Preflight memory budget gate
            if not self._preflight_ok(budget.start_pct_max):
                # One GC + short backoff, then re-check once
                _gc.collect()
                _time.sleep(0.05)
                if not self._preflight_ok(budget.start_pct_max):
                    failures += 1
                    self._log.warning(
                        "calibration_preflight_over_budget stage=%s idx=%d failures=%d/%d",
                        stage.value,
                        i,
                        failures,
                        int(budget.max_failures),
                    )
                    # Always record a checkpoint on preflight failure for observability
                    # (even if breaker will abort this stage on this attempt)
                    self._gc_and_log(stage.value, i)
                    write_checkpoint(
                        self._cfg.checkpoint_path,
                        CalibrationCheckpoint(
                            stage=stage,
                            index=i + 1,
                            results=results,
                            shortlist=None,
                            seed=None,
                        ),
                    )
                    if failures >= int(budget.max_failures):
                        self._log.error(
                            (
                                "calibration_stage_aborted stage=%s reason=circuit_breaker "
                                "failures=%d/%d"
                            ),
                            stage.value,
                            failures,
                            int(budget.max_failures),
                        )
                        break
                    continue

            self._log.info(
                "orchestrator_calling_runner stage=%s idx=%d threads=%d workers=%d",
                stage.value,
                i,
                cand.intra_threads,
                cand.num_workers,
            )
            outcome: CandidateOutcome = self._runner.run(ds, cand, int(samples), budget)
            self._log.info(
                "orchestrator_runner_returned stage=%s idx=%d ok=%s",
                stage.value,
                i,
                outcome.ok,
            )
            if outcome.ok and outcome.res is not None:
                results.append(outcome.res)
                failures = 0
            else:
                failures += 1
                err = outcome.error
                if err is not None:
                    self._log.error(
                        "calibration_candidate_failed stage=%s idx=%d kind=%s msg=%s exit=%s",
                        stage.value,
                        i,
                        err.kind,
                        err.message,
                        str(err.exit_code),
                    )
                if failures >= int(budget.max_failures):
                    self._log.error(
                        (
                            "calibration_stage_aborted stage=%s reason=circuit_breaker "
                            "failures=%d/%d"
                        ),
                        stage.value,
                        failures,
                        int(budget.max_failures),
                    )
                    break

            # Between-candidate hygiene & checkpoint
            self._gc_and_log(stage.value, i)
            write_checkpoint(
                self._cfg.checkpoint_path,
                CalibrationCheckpoint(
                    stage=stage,
                    index=i + 1,
                    results=results,
                    shortlist=None,
                    seed=None,
                ),
            )
        return results

    def run_stage_a(
        self, ds: PreprocessDataset | PreprocessSpec, cands: list[Candidate], samples: int
    ) -> list[CalibrationResult]:
        # Attempt resume
        ckpt = read_checkpoint(self._cfg.checkpoint_path)
        resume_from = 0
        prior = None
        if ckpt is not None and ckpt.stage == CalibrationStage.A:
            resume_from = int(ckpt.index)
            prior = ckpt.results
        return self._run_stage(
            stage=CalibrationStage.A,
            ds=ds,
            items=cands,
            samples=samples,
            budget=self._cfg.stage_a_budget,
            resume_from=resume_from,
            prior_results=prior,
        )

    def run_stage_b(
        self,
        ds: PreprocessDataset | PreprocessSpec,
        shortlist: list[CalibrationResult],
        samples: int,
    ) -> list[CalibrationResult]:
        # Convert shortlist results back into candidates
        cands: list[Candidate] = [
            Candidate(
                intra_threads=r.intra_threads,
                interop_threads=r.interop_threads,
                num_workers=r.num_workers,
                batch_size=r.batch_size,
            )
            for r in shortlist
        ]
        ckpt = read_checkpoint(self._cfg.checkpoint_path)
        resume_from = 0
        prior = None
        if ckpt is not None and ckpt.stage == CalibrationStage.B:
            resume_from = int(ckpt.index)
            prior = ckpt.results
        return self._run_stage(
            stage=CalibrationStage.B,
            ds=ds,
            items=cands,
            samples=samples,
            budget=self._cfg.stage_b_budget,
            resume_from=resume_from,
            prior_results=prior,
        )

    @staticmethod
    def _sort_key(r: CalibrationResult) -> tuple[float, float]:
        return (-float(r.samples_per_sec), float(r.p95_ms))

    @staticmethod
    def select_best(results: list[CalibrationResult]) -> CalibrationResult:
        # Sorted by sps desc, then p95 asc
        results_sorted = sorted(results, key=Orchestrator._sort_key)
        return results_sorted[0]


__all__ = [
    "OrchestratorConfig",
    "Orchestrator",
]
