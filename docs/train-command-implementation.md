# DiscordBot /train Command — Implementation Plan (Phase 2)

Status: design and implementation guide
Audience: handwriting-ai + DiscordBot maintainers

Purpose: add a typed, reliable `/train` command in DiscordBot that enqueues a `digits.train.v1` job to the existing worker queue (RQ → Redis), using strict contracts to prevent drift. Artifacts are uploaded by the worker to the API (v1.1 manifest only), which hot‑reloads when appropriate.

## Contracts (Stable)

- Queue: `digits`
- Target: `handwriting_ai.jobs.digits.process_train_job`
- Payload (DigitsTrainJobV1):
  - `type: "digits.train.v1"`
  - `request_id: str`
  - `user_id: int`
  - `model_id: str`
  - `epochs: int`
  - `batch_size: int`
  - `lr: float`
  - `seed: int`
  - `augment: bool`
  - `notes: str | None`
- Events channel (optional): `digits:events` (`started`/`completed`/`failed`)

Worker uploads artifacts via `POST /v1/admin/models/upload` and the API enforces:
- Manifest schema_version = `v1.1` only
- `preprocess_hash` matches service
- `model_id` matches form

## DiscordBot Changes

1) Wire the enqueuer in `src/clubbot/container.py`
- When `REDIS_URL` is present, construct `RQDigitsEnqueuer(redis_url=cfg.REDIS_URL, queue_name="digits", job_timeout_s=25200, result_ttl_s=86400, failure_ttl_s=604800, retry_max=2, retry_intervals_s=(60, 300))`.
- Pass the enqueuer to the digits cog constructor.

2) Extend `src/clubbot/cogs/digits.py`
- Update `DigitsCog.__init__(..., service: DigitService, enqueuer: DigitsEnqueuer | None = None)`; store `self.enqueuer`.
- Add `/train`:
  - Ack (defer) like `/read`, generate `request_id` via `BaseCog.new_request_id()`, call `set_request_id(request_id)`, and use `self.request_logger(request_id)`.
  - Defaults (no user args initially):
    - `model_id="mnist_resnet18_v1"`, `epochs=1`, `batch_size=256`, `lr=0.0015`, `seed=42`, `augment=True`, `notes="requested via /train"`
  - Guard: if `self.enqueuer is None`, reply ephemeral “Training is not configured.”
  - Otherwise call `self.enqueuer.enqueue_train(...)` with the payload; capture `job_id`.
  - Respond ephemeral with `request_id` and `job_id`.
  - Rate-limit if desired (reuse `RateLimiter`).

3) Tests (DiscordBot)
- Unit test for `/train`:
  - Inject a stub enqueuer that records the call and returns a fake `job_id`.
  - Assert defaults and that the message contains `req` and `job` identifiers.
  - Assert ephemeral error when enqueuer is `None`.

## Environment (Railway)

- DiscordBot service:
  - `HANDWRITING_API_URL` and `HANDWRITING_API_KEY` (for `/read`)
  - `REDIS_URL` (required for `/train` → RQ)
  - Optional: `RQ__QUEUE=digits` (enqueuer defaults already use `digits`)
- Worker service:
  - `REDIS_URL`, `RQ__QUEUE=digits`
  - `HANDWRITING_API_URL`, `HANDWRITING_API_KEY`

## End‑to‑End

1) User runs `/train`.
2) DiscordBot enqueues `digits.train.v1` to RQ queue `digits`.
3) Worker trains (MNIST) → writes v1.1 manifest → uploads to API.
4) API validates manifest (v1.1 + preprocess signature) and hot‑reloads when active.
5) `/v1/models/active` reflects the new model, `/v1/read` uses it.

## Notes on Training Modifiers

Phase 2 will add optional modifiers (default off) to the training pipeline in this repository:
- Salt/pepper noise, random dots/occlusions, Gaussian blur, erosion/dilation.
- Implemented as pre‑preprocess PIL transforms with deterministic randomness under a fixed seed.
- No change to job payload; defaults live in `TrainConfig` to avoid coupling DiscordBot to training knobs.

## Reliability & Drift Controls

- Strict types across both repos (mypy --strict; no Any, no casts, no ignores).
- Single source of truth for payload/events in handwriting‑ai; DiscordBot mirrors types.
- v1.1 manifest enforced at API; upload endpoint verifies preprocess signature.
- Structured logs with request_id at both layers for traceability.

