# Worker Volume + Training Augmentation — Verified Plan

Status: implementation guide aligned with code and design doc
Audience: worker maintainers
Principles: strict typing (mypy --strict), DRY, modular, deliberate changes only

## Overview

This guide describes how to:
- Mount a persistent `/data` volume for the Railway worker service
- Enable robust training augmentation (salt/pepper noise, random dots, optional blur/morph)

It replaces fragile line-number instructions with resilient references to functions and files, and it centers configuration in the worker runtime to prevent drift and reduce future tech debt.

---

## Verified Current State

- Augmentation implementation (complete and typed):
  - `src/handwriting_ai/training/augment.py` provides: `ensure_l_mode`, `apply_affine`, `maybe_add_noise`, `maybe_add_dots`, `maybe_blur`, `maybe_morph`.
  - `src/handwriting_ai/training/dataset.py` applies these in `PreprocessDataset.__getitem__` using knobs extracted in `_knobs_from_cfg` from a typed protocol.
  - `src/handwriting_ai/training/mnist_train.py` defines `TrainConfig` with augmentation knobs (default‑off) and `progress_every_epochs`.
- Job handler:
  - `src/handwriting_ai/jobs/digits.py` `_build_cfg` wires standard fields and basic affine augmentation; resolves `data_root` and `out_dir` via `Settings` to honor the `/data` volume. It does not pass noise/dots/blur/morph (worker runtime applies defaults when enabled).
- Worker runtime:
  - `scripts/worker._real_run_training` builds MNIST datasets at `cfg.data_root`, writes artifacts at `cfg.out_dir`, auto-sizes threads/batch, then uploads to API `/v1/admin/models/upload`.
- API:
  - `src/handwriting_ai/api/app.py` upload route writes production artifacts to `Settings.digits.model_dir` and hot‑reloads when appropriate.
- Service configuration:
  - `src/handwriting_ai/config.py` defaults to `/data` roots for app and digits model dir (overridable via env/TOML).

---

## Design Decisions (to prevent drift)

- Single source of truth for paths: Use `Settings` for both job builder and worker runtime to map to the Railway volume; avoid hardcoded constants.
- Keep job payload (`DigitsTrainJobV1`) stable; augmentation knobs are internal and default‑off in `TrainConfig`.
- Centralize runtime overrides in `scripts/worker._real_run_training` to keep unit tests isolated and avoid scattering path/knob logic.
- Maintain strict typing: no `Any`, no `typing.cast`, no `type: ignore`.

---

## Part 1: Railway Volume Setup

1) In the Railway dashboard for the worker service:
- Add a volume and mount at `/data`.

2) Configure environment variables (align with `Settings`):
- `APP__DATA_ROOT=/data`
- `APP__ARTIFACTS_ROOT=/data/artifacts`
- `REDIS_URL` and `RQ__QUEUE=digits`
- `HANDWRITING_API_URL` and `HANDWRITING_API_KEY`

Redeploy the worker service.

---

## Part 2: Integration Plan (Implemented)

Centralize path resolution and augmentation defaults in the worker runtime.

- Where: `scripts/worker.py` in `_real_run_training(cfg: TrainConfig) -> Path`
- What:
  - Derive `data_root` and `out_dir` from `Settings.load()` (which reflects `/data` volume via env/TOML) and set them on the `TrainConfig` before dataset creation.
  - When `cfg.augment` is `True` and augmentation knobs are left at defaults, set robust defaults for noise and dots (typed, no payload change).

Implementation summary (excerpt):

```python
from dataclasses import replace
from handwriting_ai.config import Settings

def _real_run_training(_: TrainConfig) -> Path:
    cfg = _
    s = Settings.load()
    data_root = (s.app.data_root / "mnist").resolve()
    out_dir = (s.app.artifacts_root / "digits" / "models").resolve()
    cfg = replace(cfg, data_root=data_root, out_dir=out_dir)

    if cfg.augment:
        cfg = replace(
            cfg,
            noise_prob=0.15,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.20,
            dots_count=3,
            dots_size_px=2,
            blur_sigma=0.0,
            morph="none",
            morph_kernel_px=1,
            progress_every_epochs=max(1, cfg.progress_every_epochs),
        )
    # proceed to dataset creation and training
```

Rationale:
- Keeps code DRY and typed; avoids duplicating path constants or knob defaults in multiple places.
- Preserves test ergonomics (tests pass their own output paths and tiny in‑memory datasets).

---

## Part 3: Job‑Side Wiring (Implemented)

In addition to the runtime wiring, the job builder now resolves paths via `Settings`:

- Where: `src/handwriting_ai/jobs/digits.py` `_build_cfg`
- Behavior:
  - Resolves `data_root = (Settings.load().app.data_root / "mnist").resolve()`
  - Resolves `out_dir = (Settings.load().app.artifacts_root / "digits" / "models").resolve()`
  - Continues to wire standard training fields and affine augmentation.
  - Does not pass noise/dots/blur/morph; worker runtime applies defaults when enabled to keep concerns centralized and typed.

---

## Part 4: Verification

Quick 1‑epoch job (enqueue from a one‑off command on the worker):

```python
python - << "PY"
import os, redis
from rq import Queue

url = os.environ["REDIS_URL"]
conn = redis.from_url(url, decode_responses=True)
q = Queue("digits", connection=conn)

payload = {
    "type": "digits.train.v1",
    "request_id": "test-volume-aug",
    "user_id": 1,
    "model_id": "mnist_resnet18_v1",
    "epochs": 1,
    "batch_size": 64,
    "lr": 0.0015,
    "seed": 42,
    "augment": True,
    "notes": "Verify /data paths and augmentation"
}

job = q.enqueue("handwriting_ai.jobs.digits.process_train_job", payload, job_timeout=3600)
print("Job ID:", job.id)
PY
```

Expected logs:
- `train_auto_config threads=... batch_size=...`
- `artifact_written_to=/data/artifacts/digits/models/mnist_resnet18_v1`
- `worker_upload_success status=200`

Signals of augmentation:
- With `augment=True`, training loss may start slightly higher; validation remains reasonable.
- Augmentation is applied in `PreprocessDataset.__getitem__` when knobs are non‑zero.

Persistence:
- Redeploying the worker should not re‑download MNIST; the dataset remains under `/data/mnist` and artifacts under `/data/artifacts`.

---

## Part 5: Production Run Example

```python
python - << "PY"
import os, redis
from rq import Queue

url = os.environ["REDIS_URL"]
conn = redis.from_url(url, decode_responses=True)
q = Queue("digits", connection=conn)

payload = {
    "type": "digits.train.v1",
    "request_id": "prod-50ep-noise-robust",
    "user_id": 1,
    "model_id": "mnist_resnet18_v1",
    "epochs": 50,
    "batch_size": 64,
    "lr": 0.0015,
    "seed": 42,
    "augment": True,
    "notes": "Production training with noise robustness"
}

job = q.enqueue("handwriting_ai.jobs.digits.process_train_job", payload, job_timeout=86400)
print("Job ID:", job.id)
PY
```

Expected results:
- Robust accuracy near baseline, potentially slightly lower due to harder data.
- All artifacts and MNIST content persist on `/data`.

---

## Checklist (Before Production)

- [ ] Worker has `/data` volume mounted
- [ ] Env set: `APP__DATA_ROOT`, `APP__ARTIFACTS_ROOT`, `REDIS_URL`, `RQ__QUEUE`, `HANDWRITING_API_URL`, `HANDWRITING_API_KEY`
- [ ] 1‑epoch test job completed successfully
- [ ] Logs show `/data/...` paths, not relative paths
- [ ] MNIST persists across redeploys
- [ ] Augmentation knobs active when `augment=True`

---

## Troubleshooting

MNIST re‑downloads each run
- Volume missing or mount path not `/data`; or runtime isn't using `Settings` paths.

Augmentation appears off
- Ensure payload sets `augment=True`.
- Confirm runtime injects non‑zero knobs (preferred path) or job builder passes them.

Upload failures
- Verify `HANDWRITING_API_URL` and `HANDWRITING_API_KEY` are set and valid.
- Confirm artifacts written under `/data/artifacts/digits/models/...` before upload.

---

## References (stable, no line numbers)

- Augmentation: `src/handwriting_ai/training/augment.py`
- Dataset wrapper: `src/handwriting_ai/training/dataset.py` (`PreprocessDataset`, `_knobs_from_cfg`)
- Train config + loop: `src/handwriting_ai/training/mnist_train.py` (`TrainConfig`, `train_with_config`)
- Job handler: `src/handwriting_ai/jobs/digits.py` (`_build_cfg`, `process_train_job`)
- Worker runtime: `scripts/worker.py` (`_real_run_training`, upload helpers)
- API upload: `src/handwriting_ai/api/app.py` (admin upload route)
- Multi‑phase design doc: `docs/Digits-Service-Plan.md`

---

## Why This Approach Reduces Tech Debt

- Centralized runtime wiring avoids duplicated constants and keeps tests isolated.
- Strong, frozen dataclasses + Protocols + TypedDicts across the pipeline preserve strict typing guarantees.
- Stable references (functions/files) prevent doc drift and line‑number brittleness.
