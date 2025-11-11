from __future__ import annotations

import gc as _gc
import multiprocessing as _mp
import os as _os
import time as _time
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Protocol

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.measure import CalibrationResult, _measure_candidate
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.safety import MemoryGuardConfig, set_memory_guard_config


@dataclass(frozen=True)
class CandidateError:
    kind: str  # "timeout" | "oom" | "runtime"
    message: str
    exit_code: int | None


@dataclass(frozen=True)
class CandidateOutcome:
    ok: bool
    res: CalibrationResult | None
    error: CandidateError | None


@dataclass(frozen=True)
class BudgetConfig:
    start_pct_max: float
    abort_pct: float
    timeout_s: float
    max_failures: int


class CandidateRunner(Protocol):
    def run(  # pragma: no cover - signature line not counted reliably by coverage on some setups
        self,
        ds: PreprocessDataset,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome: ...


def _encode_result_kv(res: CalibrationResult) -> list[str]:
    """Encode a CalibrationResult into key=value lines for IPC."""
    inter_str = "" if res.interop_threads is None else str(int(res.interop_threads))
    return [
        "ok=1",
        f"intra_threads={int(res.intra_threads)}",
        f"interop_threads={inter_str}",
        f"num_workers={int(res.num_workers)}",
        f"batch_size={int(res.batch_size)}",
        f"samples_per_sec={float(res.samples_per_sec)}",
        f"p95_ms={float(res.p95_ms)}",
    ]


def _write_kv(out_path: str, lines: list[str]) -> None:
    """Write key=value lines to a file (UTF-8)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _emit_result_file(out_path: str, res: CalibrationResult) -> None:
    """Encode and write a result file for IPC."""
    _write_kv(out_path, _encode_result_kv(res))


def _child_entry(
    out_path: str,
    ds: PreprocessDataset,
    cand: Candidate,
    samples: int,
    abort_pct: float,
) -> None:
    import logging as _logging
    import time as _time

    log = _logging.getLogger("handwriting_ai")
    start_entry = _time.perf_counter()
    log.info(
        "calibration_child_started pid=%d threads=%d workers=%d bs=%d",
        _os.getpid(),
        cand.intra_threads,
        cand.num_workers,
        cand.batch_size,
    )
    try:
        # Configure memory guard inside the child for calibration attempts.
        start_guard = _time.perf_counter()
        set_memory_guard_config(
            MemoryGuardConfig(
                enabled=True,
                threshold_percent=float(abort_pct),
                required_consecutive=3,
            )
        )
        guard_elapsed = _time.perf_counter() - start_guard
        log.info("calibration_child_guard_set elapsed_ms=%.1f", guard_elapsed * 1000)

        start_measure = _time.perf_counter()
        res = _measure_candidate(ds, cand, samples)
        measure_elapsed = _time.perf_counter() - start_measure
        log.info("calibration_child_measure_complete elapsed_s=%.1f", measure_elapsed)

        # Manual KV encoding of result (executes in child process)
        _emit_result_file(out_path, res)  # pragma: no cover (subprocess)

        total_elapsed = _time.perf_counter() - start_entry
        log.info("calibration_child_complete total_s=%.1f", total_elapsed)
    finally:
        # Ensure prompt teardown
        _gc.collect()


class SubprocessRunner:
    def __init__(self) -> None:
        # Use spawn for consistent behavior across OSes
        self._ctx = _mp.get_context("spawn")

    def run(
        self,
        ds: PreprocessDataset,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        import logging as _logging
        import tempfile as _tmp

        log = _logging.getLogger("handwriting_ai")
        out_dir = _tmp.mkdtemp(prefix="calib_child_")
        out_path = _os.path.join(out_dir, "result.txt")

        spawn_start = _time.perf_counter()
        proc = self._ctx.Process(
            target=_child_entry,
            args=(out_path, ds, cand, int(samples), float(budget.abort_pct)),
        )
        start = _time.perf_counter()
        proc.start()
        spawn_elapsed = _time.perf_counter() - spawn_start
        log.info(
            "calibration_parent_spawned threads=%d workers=%d bs=%d spawn_ms=%.1f timeout_s=%.1f",
            cand.intra_threads,
            cand.num_workers,
            cand.batch_size,
            spawn_elapsed * 1000,
            float(budget.timeout_s),
        )

        try:
            return self._wait_for_outcome(proc, out_path, start, float(budget.timeout_s))
        finally:
            if proc.is_alive():
                with suppress(Exception):
                    proc.kill()
                with suppress(Exception):
                    proc.join(1.0)
            _gc.collect()

    def _wait_for_outcome(
        self,
        proc: BaseProcess,
        out_path: str,
        start: float,
        timeout_s: float,
    ) -> CandidateOutcome:
        # Poll for child result (file) with incremental waits
        while proc.is_alive():
            outcome = self._try_read_result(out_path, exited=False, exit_code=None)
            if outcome is not None:
                return outcome
            if (_time.perf_counter() - start) >= timeout_s:
                # Timeout: terminate and mark
                with suppress(Exception):
                    proc.terminate()
                with suppress(Exception):
                    proc.join(2.0)
                return CandidateOutcome(
                    ok=False,
                    res=None,
                    error=CandidateError(
                        kind="timeout",
                        message="candidate timed out",
                        exit_code=None,
                    ),
                )
            _time.sleep(0.01)

        # Process not alive: try to read any pending outcome
        outcome2 = self._try_read_result(out_path, exited=True, exit_code=proc.exitcode)
        if outcome2 is not None:
            return outcome2

        # Determine exit condition
        code = proc.exitcode
        if code in (-9, 137):
            return CandidateOutcome(
                ok=False,
                res=None,
                error=CandidateError(
                    kind="oom", message="child killed (possible OOM)", exit_code=code
                ),
            )
        return CandidateOutcome(
            ok=False,
            res=None,
            error=CandidateError(
                kind="runtime", message=f"child exited code={code}", exit_code=code
            ),
        )

    @staticmethod
    def _try_read_result(
        out_path: str, *, exited: bool, exit_code: int | None
    ) -> CandidateOutcome | None:
        if not _os.path.exists(out_path):
            return None
        if not (_os.path.exists(out_path) and _os.access(out_path, _os.R_OK)):
            return None
        with open(out_path, encoding="utf-8") as f:
            content = f.read()
        # Parse simple key=value lines
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        data: dict[str, str] = {}
        for ln in lines:
            if "=" not in ln:
                return None
            k, v = ln.split("=", 1)
            data[k] = v

        if data.get("ok") == "1":
            # Required fields
            intra = int(data.get("intra_threads", "0"))
            inter_raw = data.get("interop_threads", "")
            inter = int(inter_raw) if inter_raw != "" else None
            nworkers = int(data.get("num_workers", "0"))
            bs = int(data.get("batch_size", "1"))
            sps = float(data.get("samples_per_sec", "0.0"))
            p95 = float(data.get("p95_ms", "0.0"))
            res = CalibrationResult(
                intra_threads=intra,
                interop_threads=inter,
                num_workers=nworkers,
                batch_size=bs,
                samples_per_sec=sps,
                p95_ms=p95,
            )
            return CandidateOutcome(ok=True, res=res, error=None)

        if data.get("ok") == "0":
            msg = data.get("error_message", "")
            return CandidateOutcome(
                ok=False,
                res=None,
                error=CandidateError(
                    kind="runtime", message=msg, exit_code=exit_code if exited else None
                ),
            )
        return None


__all__ = [
    "CandidateError",
    "CandidateOutcome",
    "BudgetConfig",
    "CandidateRunner",
    "SubprocessRunner",
]
