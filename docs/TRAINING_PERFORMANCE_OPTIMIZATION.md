# Training Performance Optimization - Design and Implementation Plan

Status: ready to implement
Audience: handwriting-ai maintainers
Principles: type safety, DRY, modular, observable, zero quick fixes

## Executive Summary

- Current: ~16 minutes/epoch on 60k samples, batch_size=256
- Target: 2–5 minutes/epoch (hardware-dependent, 4–10x faster)
- Symptoms: training loop stalls on data prep; CPU underutilized; long feedback loop for model iteration

## Root Causes

- Synchronous DataLoader: `src/handwriting_ai/training/dataset.py` uses `num_workers=0` in `make_loaders`.
- Thread contention & duplication: ad-hoc detection in `scripts/worker.py` (`_detect_cpu_threads`, `_mem_limit_bytes`, `_decide_batch_cap`), not integrated with training.
- Configuration validation: no single place to validate threads, batch caps, DataLoader workers, or to log decisions.

## Solution Overview

- Resource detection module (new): centralize detection and derived limits.
  - `ResourceLimits` (frozen): `cpu_cores`, `memory_bytes|None`, `optimal_threads`, `optimal_workers`, `max_batch_size|None`.
  - Implement cgroup-aware detection (v2 then v1) with fallbacks; compute optimal threads/workers conservatively; compute memory-based batch cap.
  - Log detection once at INFO with key=value pairs.

- DataLoader configuration (explicit): separate from training knobs.
  - `DataLoaderConfig` (frozen): `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`.
  - Validation: `batch_size>0`, `num_workers>=0`, `prefetch_factor>=0`, `persistent_workers` requires `num_workers>0`.
  - `make_loaders(...)` accepts `loader_cfg` (keeps API stable if optional) and applies settings uniformly for train/test; only set `prefetch_factor`/`persistent_workers` when `num_workers>0`.

- Training integration: resource-aware, observable, and type-safe.
  - In `src/handwriting_ai/training/mnist_train.py`:
    - Detect resources once; cap effective batch size if `max_batch_size` present.
    - Configure PyTorch threads (`intra`, `interop` ~ half) from `optimal_threads` when `cfg.threads<=0`.
    - Build `DataLoaderConfig` using `optimal_workers`, `prefetch_factor=2`, `pin_memory=False` (CPU); log at INFO.
    - Create loaders via `make_loaders(train_base, test_base, cfg, loader_cfg)`.

- Worker refactor: delegate optimization to training.
  - In `scripts/worker.py`: remove duplicate `_detect_cpu_threads/_mem_limit_bytes/_decide_batch_cap` and stop overriding cfg; keep path resolution and augmentation defaults; call `train_with_config`.

## Empirical Auto-Calibration (No Guesswork)

Replace heuristics with a fast, deterministic preflight that measures only the input pipeline (no model training) and persists the result.

- Scope: CPU only. Runs before training or when the environment changes.
- Candidates (bounded by detected CPU and memory):
  - Threads (intra-op): `{max(1, cpu_cores//2), cpu_cores}` (clamped on tiny machines to `{1,2}` for 2 vCPU). Inter-op := `max(1, intra//2)`.
  - DataLoader workers: `{0,1,2}` but never `> cpu_cores//2`, and default `0` on =2 cores unless augmentation is heavy.
  - Batch size: start from requested; cap by memory (e.g., 64–128 on 1 GB); probe down on OOM.
- Procedure (typed, reproducible):
  1) Warm up each candidate: initialize DataLoader and fetch 1 batch.
  2) Measure K fetches (e.g., K=8–16) of `next(loader_iter)` without model forward/backward.
  3) Record samples/sec and p95 batch time; rank by samples/sec, tie-break by p95.
  4) Persist the winner to `artifacts/calibration.json` with signature `{cpu_cores, mem_bytes, os, py, torch}` and TTL (e.g., 7 days).
- Safety and time budget:
  - Catch `MemoryError`/timeouts; back off batch size and demote the candidate.
  - Early stop candidates =15% slower than current best after a few batches.
  - Typical wall time: 2–5 seconds on small machines; scales modestly with cores.
- Logging (INFO): one line per candidate with `key=value` metrics; one line for the winner.
- Overrides: `--calibrate/--no-calibrate`, `--calibration-samples`, `--force-calibration`.

This removes guesswork and avoids any “train to tune” overhead. Decisions are measured, logged, and reused until the environment changes.

## Design Principles (enforced)

- Type safety: no `Any`, no `typing.cast`, no `# type: ignore` anywhere.
- Frozen dataclasses; validate at construction; no magic numbers (document constants).
- DRY: one resource detection module; one DataLoader configuration point.
- Observability: log resource limits and effective training/DataLoader choices with structured key=value.
- No quick fixes: no env hacks, no hard-coded thread counts.

## Implementation Steps (condensed)

1) Resources module
- Add `src/handwriting_ai/training/resources.py` with detection + computations and INFO logging.
- Unit tests cover: cgroups fallbacks, optimal threads/workers, batch caps, immutability.

2) DataLoader configuration
- Add `DataLoaderConfig` into `dataset.py`; update `make_loaders` to accept `loader_cfg` (optionally) and apply.
- Unit tests cover: validation paths, immutability, integration (prefetch/persistent only when `num_workers>0`).

3) Training integration
- In `mnist_train.py`: detect resources, configure threads, cap batch size, build `DataLoaderConfig`, log, create loaders.
- Optional: if `threadpoolctl` is available, wrap training in a single `threadpool_limits(limits=intra)` context to align MKL/OpenBLAS/OMP pools with ATen and log detected/limited pools; skip silently if not installed.
- Tests cover: resource use (via monkeypatch), logs, thread settings chosen.

4) Worker refactor
- Remove ad-hoc detection; call `train_with_config` (which now optimizes); keep path resolution and augmentation defaults.
- Tests cover: unchanged behavior for paths/aug defaults; no duplicate tuning.

## Validation & Quality Gates

- Typing & guards
  - Mypy clean; all new and existing code typed.
  - Extend `scripts/guards/typing_rules.py` to include `scripts/`; zero `Any`, zero `cast`, zero `# type: ignore` across repo.

- Tests & checks
  - `make check` runs ruff, mypy, guards, yaml lint, and pytest.
  - Add focused tests only; maintain high coverage (no net decrease).

- Manual validation
  - `make train-long` once integrated; expect logs:
    - `resource_limits cpu_cores=... memory_mb=... optimal_threads=... optimal_workers=... max_batch_size=...`
    - `training_config device=cpu threads_intra=... threads_interop=... batch_size=...`
    - `dataloader_config num_workers=... persistent_workers=... prefetch_factor=...`
  - Expect 2–5 min/epoch on typical CPU (depends on host).

## Deployment

- No config changes required; deploy as usual; monitor logs for resource/dataloader lines.
- Rollback: revert service; no stateful changes.

## Alignment With Current Codebase

- Synchronous DataLoader confirmed: `num_workers=0` in `src/handwriting_ai/training/dataset.py` `make_loaders`.
- Duplication confirmed: thread/memory logic in `scripts/worker.py`.
- Training does not yet detect resources; only applies threads if `cfg.threads>0`.
- Guard scope: currently `src/` and `tests/`; extend to `scripts/`. There is a `# type: ignore[...]` in `scripts/prestart.py` to remove during implementation.
- Makefile: `train-long` is correct target.

## Out of Scope (for focus)

- GPU-specific tuning; mixed precision; distributed training; disk-cached preprocessing; dynamic batch size at runtime.

## Summary

- Parallelize data prep with optimal workers to eliminate stalls.
- Cap and structure PyTorch threading to avoid contention.
- Centralize resource detection; centralize DataLoader config; add clear logs.
- Enforce strict typing and repo-wide guard checks.
- Expected improvement: 4–10x speedup (16min ? 2–5min/epoch) with no tech debt.
