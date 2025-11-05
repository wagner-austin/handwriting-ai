# Rich Training Events — Design and Implementation Plan (v1)

Status: ready to implement (code audited and aligned)
Audience: handwriting-ai + DiscordBot maintainers
Principles: strict typing, DRY, modular, observable, minimal runtime overhead

## Goals

- Emit rich, structured training events from the worker while keeping training hot path fast.
- Provide a stable, versioned event contract consumed by DiscordBot for live progress updates.
- Avoid application drift: single source for event schemas in code, covered by tests.
- Maintain strict typing end-to-end (Python TypedDict, TypeScript discriminated unions). No Any, no casts.

## Current State (Audited)

- Event publishing exists in `src/handwriting_ai/jobs/digits.py` using a `Publisher` protocol, channel from env (`DIGITS_EVENTS_CHANNEL`, default `digits:events`).
  - Events today: `started`, `progress` (epoch-level), `completed`, `failed`. Payloads are TypedDicts and JSON-encoded.
- Training loop logs detailed batch metrics to the logger (not events): see `src/handwriting_ai/training/loops.py` (info logs every ~10 batches) and epoch summaries/emitter in `src/handwriting_ai/training/mnist_train.py`.
- Worker runtime (`scripts/worker.py`) publishes no events around upload/prune (only logs). Upload and pruning are now robust and typed.
- Tests exist for job events and progress emitter wiring. Strict typing and 100% coverage are enforced.

Conclusion: A solid baseline exists. We will extend events without breaking existing behavior.

## Event Model (v1)

Topic: Redis Pub/Sub channel `digits:events` (configurable). JSON messages with `type` discriminator.

Common fields (all events):
- `type: string` — one of types below
- `request_id: string` — correlates to Discord request
- `user_id: int` — originating user
- `model_id: string`
- `run_id: string | null` — set once known (after artifacts start or manifest has a run id)
- `ts: string` — ISO 8601 timestamp (UTC)

Event types (additive to existing):
- `digits.train.started.v1`
  - `{ total_epochs: int }`
- `digits.train.batch.v1` (optional, rate-limited)
  - `{ epoch: int, total_epochs: int, batch: int, total_batches: int, batch_loss: float, batch_acc: float, avg_loss: float, samples_per_sec: float }`
- `digits.train.epoch.v1`
  - `{ epoch: int, total_epochs: int, train_loss: float, val_acc: float, time_s: float }`
- `digits.train.best.v1`
  - `{ epoch: int, val_acc: float }`
- `digits.train.artifact.v1`
  - `{ path: string }`
- `digits.train.upload.v1`
  - `{ status: int, model_bytes: int, manifest_bytes: int }`
- `digits.train.prune.v1`
  - `{ deleted_count: int }`
- `digits.train.completed.v1`
  - `{ val_acc: float }`
- `digits.train.failed.v1`
  - `{ error_kind: 'user' | 'system', message: string }`

Notes:
- Use `.v1` suffix to allow future evolution without ambiguity.
- Existing compact events remain supported; DiscordBot should handle both, preferring `.v1`.

## Code Changes (handwriting-ai)

1) Event schemas (strict typing)
   - New module `src/handwriting_ai/events/digits.py` with TypedDicts for all events and `encode_event()`.
   - Reuse `Publisher` protocol from `jobs.digits` or move to a shared module to avoid duplication.

2) Progress emitter extension
   - Option A (preferred): add a new optional BatchProgressEmitter protocol with `emit_batch(...)` called from `loops.train_epoch` on the existing cadence (every 10 batches).
   - Option B: extend existing `ProgressEmitter` with an `emit_batch(...)` default no-op via duck typing. Keep `emit(...)` unchanged for epoch.
   - Configuration:
     - `EVENTS__BATCH_EVERY: int = 10` (0 disables batch events)
     - `EVENTS__ENABLED: bool = true` (global kill switch)

3) Jobs: publish enriched started/completed/failed using new schemas
   - Update `src/handwriting_ai/jobs/digits.py` to include `ts` and `run_id` when available.

4) Training: publish epoch/best events via emitter bridge
   - In `mnist_train.train_with_config`, after computing `val_acc`, emit `epoch.v1` and when better, `best.v1`.
   - Derive `run_id` from `write_artifacts` manifest (already included) and include it in subsequent events.

5) Worker: publish upload/prune events
   - In `scripts/worker.py` `_maybe_upload_artifacts`: publish `upload.v1` (status and sizes) and `prune.v1` (deleted_count).
   - Never block training on publish failures; log and continue.

6) Settings
   - Add `events` section or reuse existing `DIGITS_EVENTS_CHANNEL`.
   - New envs: `EVENTS__ENABLED`, `EVENTS__BATCH_EVERY`.
   - Keep typing strict (no Any), add to `Settings` and loaders.

7) Tests
   - Unit tests for each event producer, monkeypatching `Publisher` to capture messages.
   - Assert strict schemas, types, and presence of discriminators.
   - Ensure batch cadence config disables noisy events in tests by default.

## DiscordBot Changes

1) Add a Redis subscriber client (httpx and redis are already in use in many bots)
   - Subscribe to `digits:events`.
   - Parse events using zod schemas (strict) with discriminated union on `type`.
   - Update Discord messages in-place keyed by `{request_id, run_id}`:
     - Started: send initial embed
     - Batch: update progress bar and batch stats every few updates (throttle to 1–2s)
     - Epoch: append epoch summary and current `val_acc`
     - Best: highlight new best
     - Upload/Prune: note artifacts uploaded and snapshots pruned
     - Completed/Failed: finalize message

2) Types (TypeScript)
```ts
type BaseEvt = {
  type: string;
  request_id: string;
  user_id: number;
  model_id: string;
  run_id: string | null;
  ts: string;
};

type StartedV1 = BaseEvt & { type: 'digits.train.started.v1'; total_epochs: number };
type BatchV1 = BaseEvt & { type: 'digits.train.batch.v1'; epoch: number; total_epochs: number; batch: number; total_batches: number; batch_loss: number; batch_acc: number; avg_loss: number; samples_per_sec: number };
type EpochV1 = BaseEvt & { type: 'digits.train.epoch.v1'; epoch: number; total_epochs: number; train_loss: number; val_acc: number; time_s: number };
type BestV1 = BaseEvt & { type: 'digits.train.best.v1'; epoch: number; val_acc: number };
type ArtifactV1 = BaseEvt & { type: 'digits.train.artifact.v1'; path: string };
type UploadV1 = BaseEvt & { type: 'digits.train.upload.v1'; status: number; model_bytes: number; manifest_bytes: number };
type PruneV1 = BaseEvt & { type: 'digits.train.prune.v1'; deleted_count: number };
type CompletedV1 = BaseEvt & { type: 'digits.train.completed.v1'; val_acc: number };
type FailedV1 = BaseEvt & { type: 'digits.train.failed.v1'; error_kind: 'user' | 'system'; message: string };

type DigitsEvt = StartedV1 | BatchV1 | EpochV1 | BestV1 | ArtifactV1 | UploadV1 | PruneV1 | CompletedV1 | FailedV1;
```

3) UX
   - Use a single Discord message per train job, edited in place.
   - Include a simple text-based progress bar; throttle batch updates.
   - Surface API error codes/messages (already strict) for training failures.

## Reliability & Performance

- Publishing is best-effort and non-blocking; failures are logged and never raise into the training loop.
- Batch events can be disabled or rate-limited to avoid chat noise.
- Event schemas are versioned; DiscordBot uses discriminated unions to avoid brittle parsing.
- All code paths covered by unit tests; mypy --strict; ruff clean; guard checks enforced (no silent excepts).

## Rollout Plan

1) Implement event schemas and producer wiring in handwriting-ai.
2) Add tests to maintain 100% coverage; run `make check`.
3) Implement DiscordBot subscriber with zod schemas and throttled updates.
4) Feature flag batch events off by default in production; enable gradually.
5) Monitor logs and Discord UX; tune `EVENTS__BATCH_EVERY` as needed.

## References

- Job events: `src/handwriting_ai/jobs/digits.py`
- Training loops: `src/handwriting_ai/training/loops.py`
- Training driver: `src/handwriting_ai/training/mnist_train.py`
- Worker runtime: `scripts/worker.py`

