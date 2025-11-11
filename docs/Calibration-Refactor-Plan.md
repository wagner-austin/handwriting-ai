# Calibration Refactor Plan: Isolation, Budgets, Checkpoints

Status: Draft (proposed)
Owner: Training/Calibration subsystem
Targets: 1 GB containers (≈953 MB) and larger

Compatibility Policy: No fallbacks, no legacy paths, no back-compat shims.

## Goals
- Reliability under tight memory (≤1 GB cgroups) with no OOM kills.
- Deterministic, observable calibration with explicit isolation and budgets.
- Clear boundaries (orchestrator/runner) and typed contracts (no Any/casts/ignores).
- Checkpoints to resume safely; circuit breaker to abort gracefully.
- Public API may change where required; update all call sites within the repo accordingly. Cache format remains unchanged unless explicitly versioned.
- Full tests for new modules; `make check` passes (ruff + mypy --strict + pytest with coverage).

## Non‑Goals
- Changing the measurement logic itself (e.g., replacing `_measure_candidate`).
- Altering the on‑disk cache schema or train runtime.

## Architecture Overview
- Orchestrator (parent): Schedules candidates, enforces memory/time budgets, writes checkpoints, and applies circuit breaking. Logs between‑candidate snapshots.
- Runner (child): Executes one candidate in isolation.
  - SubprocessRunner (only): Fresh process per candidate; strict timeout; OOM/kill detection. No in-process fallback. If isolation cannot be established, calibration fails fast.
- Checkpointing: Persist minimal progress to resume mid‑stage without re‑measuring prior candidates.
- Budgeting: Stage‑aware thresholds (start gate, abort threshold, timeout) tuned to cgroup limit.
- Circuit Breaker: Stop a stage after repeated failures (timeout/OOM/over‑budget) and select the best available configuration or a conservative default.

## Modules & Types (strict typing, no Any)

File: `src/handwriting_ai/training/calibration/orchestrator.py`
- enum CalibrationStage: { A, B }
- @dataclass CandidateSpec: re‑use `Candidate` from `candidates.py` (no schema drift)
- @dataclass MeasurementMetrics:
  - samples_per_sec: float
  - p95_ms: float
  - peak_mem_pct: float
- @dataclass CandidateError:
  - kind: Literal["timeout", "oom", "runtime"]
  - message: str
  - exit_code: int | None
- @dataclass CandidateOutcome:
  - ok: bool
  - res: CalibrationResult | None
  - error: CandidateError | None
- @dataclass BudgetConfig:
  - start_pct_max: float      # pre‑flight: must be ≤ this before starting
  - abort_pct: float          # if child reports ≥ this, count a failure
  - timeout_s: float          # per‑candidate wall timeout
  - max_failures: int         # circuit breaker threshold
- @dataclass CircuitBreaker:
  - failures: int
  - max_failures: int
- @dataclass CalibrationCheckpoint:
  - stage: CalibrationStage
  - index: int
  - results: list[CalibrationResult]
  - shortlist: list[CalibrationResult] | None
  - seed: int | None

Interfaces
- class CandidateRunner(Protocol):
  - def run(self, ds: PreprocessDataset, cand: Candidate, samples: int, budget: BudgetConfig) -> CandidateOutcome: ...
- class Orchestrator:
  - __init__(..., runner: CandidateRunner, budget: BudgetConfig) -> None
  - run_stage_a(self, ds, cands, samples) -> list[CalibrationResult]
  - run_stage_b(self, ds, shortlist, samples) -> list[CalibrationResult]
  - select_best(self, results) -> CalibrationResult

File: `src/handwriting_ai/training/calibration/runner.py`
- class SubprocessRunner(CandidateRunner):
  - Uses `multiprocessing.get_context("spawn")` for a clean child process.
  - Child entrypoint executes `_measure_candidate(ds, cand, samples)`, returns a minimized payload through a Pipe.
  - Parent enforces timeout; on timeout → terminate/kill → return CandidateError(kind="timeout").
  - Map exit codes/signals to `oom` when indicative (e.g., SIGKILL 9 / exit code 137 on Linux); else `runtime`.
  - Parent collects GC and logs between‑candidate snapshot.
  - Dataset must be serializable for spawn. Implement explicit `__getstate__/__setstate__` where needed (e.g., dataset wrappers) instead of falling back.

File: `src/handwriting_ai/training/calibration/checkpoint.py`
- read_checkpoint(path: Path) -> CalibrationCheckpoint | None
- write_checkpoint(path: Path, ckpt: CalibrationCheckpoint) -> None
- JSON encode/decode with typed helpers only (no dynamic dict[Any, Any]).

Refactor: `src/handwriting_ai/training/calibration/calibrator.py`
- Replace current implementation with orchestrator-driven flow (no legacy code retained).
- Build `Orchestrator` with `SubprocessRunner` and stage budgets computed from `ResourceLimits` + `compute_memory_guard_config()`.
- Stage A: run cands; sort + shortlist.
- Stage B: run shortlist; select best; write cache; clear checkpoint on success.

## Budgeting & Thresholds
- Derive baseline thresholds from `compute_memory_guard_config(memory_bytes)`:
  - <1 GB → threshold ≈ 80–85%
  - 1–2 GB → ≈ 85%
  - 2–4 GB → ≈ 88%
  - ≥4 GB → ≈ 92%
- Suggested stage budgets for <1 GB:
  - Stage A: start_pct_max=80.0, abort_pct=85.0, timeout_s=45, max_failures=2
  - Stage B: start_pct_max=83.0, abort_pct=88.0, timeout_s=60, max_failures=2
- Pre‑flight gate: if current cgroup usage% > start_pct_max, run one `gc.collect()`, re‑check once; still high → short back‑off sleep; still high → increment breaker and skip candidate (or abort stage if breaker trips).
- Candidate ordering bias: test lower workers/threads first to maximize headroom in constrained memory.

## Control Flow (high level)
```
# Stage A
breaker = CircuitBreaker(failures=0, max_failures=budget.max_failures)
results: list[CalibrationResult] = []
for idx, cand in enumerate(cands):
    if over_start_gate():
        if breaker_bump_and_maybe_abort(): break
        continue
    outcome = runner.run(ds, cand, samples, budget)
    if outcome.ok and outcome.res is not None:
        results.append(outcome.res)
    else:
        breaker.failures += 1
        log_candidate_error(...)
        if breaker.failures >= breaker.max_failures:
            log_stage_abort(...)
            break
    between_candidates_gc_and_log()
# shortlist and Stage B similar pattern; choose best
```

## Failure Modes & Handling
- Timeout → error(kind="timeout") and breaker++.
- SIGKILL/exit 137 (Linux) → error(kind="oom") and breaker++.
- Child returns `exceeded=True`/peak above `abort_pct` → treat as failure and breaker++.
- On breaker trip: stop stage; use best known result or conservative fallback (threads=1, workers=0, batch_size small).

## Checkpointing
- Contents: stage, next index, completed results, shortlist (when moving from A→B), optional seed.
- Write after each candidate outcome.
- On calibrate start: if valid checkpoint matches signature and TTL, resume from it; else discard.
- On success: delete checkpoint.

## Memory Hygiene Guarantees
- Subprocess isolation ensures model/optimizer/DataLoader allocations are reclaimed by OS after each child exit.
- Parent process performs `gc.collect()` and logs snapshots between candidates for observability.
- Dataset reuse: acceptable in parent (stateless wrapper); if pickling fails, orchestrator logs and falls back to InProcessRunner for that environment, keeping behavior predictable.

## Logging (consistent with existing style)
- `calibration_between_candidates_gc stage=A|B idx=%d mem_pct=%.1f cgroup_mb=%d main_rss_mb=%d workers_rss_mb=%d anon_mb=%d file_mb=%d kernel_mb=%d slab_mb=%d`
- `calibration_candidate_timeout stage=%s idx=%d seconds=%.1f`
- `calibration_candidate_oom stage=%s idx=%d exit=%d`
- `calibration_stage_aborted stage=%s reason=circuit_breaker failures=%d/%d`
- Final report remains unchanged.

## Testing Strategy
- Unit (orchestrator): happy path ordering, pre‑flight gates invoked, breaker trip after configured failures, checkpoint resume semantics, budget gating behavior when over threshold.
- Unit (runner): SubprocessRunner polling/timeout code paths (simulated child behavior via small entrypoints); serialization enforcement for dataset wrappers.
- Integration: SubprocessRunner end‑to‑end for a small FakeMNIST dataset; timeout path (sleeping child); simulated OOM (child exits with 137) → mapped to `oom`.
- Contract tests: serialization of checkpoint; no Any/casts/ignores in new code; strict mypy and ruff pass.
- Coverage: 100% statements and branches for new modules; repository `make check` remains green.

## Migration Plan
1) Add orchestrator, runner (subprocess‑only), checkpoint modules with full typing and tests.
2) Replace calibrator implementation to delegate to orchestrator exclusively; update any internal call sites accordingly (no dual paths).
3) Validate on 953 MB container; confirm logs show per‑candidate isolation and bounded usage.
4) Remove ad‑hoc cleanup now superseded by isolation.

## Acceptance Criteria
- No OOM kills during calibration on 1 GB containers.
- Repeatable results; clear logs of between‑candidate GC and child isolation.
- Circuit breaker aborts gracefully with best known or conservative fallback.
- Checkpointing allows resume mid‑stage with no duplicate work.
- New code has zero Any/casts/ignores; mypy strict clean; ruff clean.
- Tests for new modules at 100% coverage (statements + branches); `make check` passes.

## Open Questions / Follow‑ups
- Do we want per‑candidate dynamic timeouts based on dataset size and batch size hint?
- Should we persist memory diagnostics per candidate in the cache for future runs (telemetry)?
- Consider optional per‑candidate `process.setrlimit` (when available) to hard‑cap memory.

---
This document is the single reference for the calibration refactor. Update alongside implementation PRs and keep `make check` green at each step.
