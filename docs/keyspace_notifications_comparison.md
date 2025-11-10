# Redis Keyspace Notifications vs Polling — Audit and Migration Plan (No Fallback)

This document audits the current codebase and specifies a complete migration from polling to Redis keyspace notifications for RQ registries. The migration is strict: no fallbacks, no back-compat toggles, and removal of legacy polling code. All changes must preserve our strict typing (no Any/casts/ignores), DRY structure, and 100% statement/branch coverage under `make check`.

## Repository Audit (Current State)

- Polling watcher: `src/handwriting_ai/jobs/watcher/watcher.py:21` implements `FailureWatcher` which scans RQ registries every `poll_interval_s` seconds and publishes typed failure events. It is wired via `scripts/rq_failure_watcher.py:1` and `src/handwriting_ai/jobs/watcher/runner.py:1`.
- Integration boundaries are abstracted through `WatcherPorts` for redis/rq access: `src/handwriting_ai/jobs/watcher/ports.py:1` and `src/handwriting_ai/jobs/watcher/adapters.py:1`.
- Failure heuristics and payload extraction are typed helpers: `src/handwriting_ai/jobs/watcher/logic.py:1`.
- Publisher and dedupe store are small, typed components: `src/handwriting_ai/jobs/watcher/publisher.py:1`, `src/handwriting_ai/jobs/watcher/store.py:1`.
- There is no event‑driven watcher in the repo today. A demo exists for Upstash keyspace notifications: `scripts/test_keyspace_notifications.py:1`.
- Tests thoroughly cover polling behavior: `tests/test_failure_watcher.py:1` and companions; CI gates enforce strict typing and coverage via `make check` (ruff + mypy strict + pytest coverage).

Conclusion: We currently poll. We will replace polling with an event-driven watcher, then remove polling code and its tests.

## Why Keyspace Notifications

- Idle cost and noise: polling touches three registries per scan (failed/started/canceled), even when nothing changes.
- Event-driven flips the model: subscribe once; react only when Redis reports a ZSET change on the specific registry keys.

## Target Design (Event-Driven Only)

New synchronous watcher consumes keyspace notifications for the three RQ registries: `failed`, `started`, and `canceled` for the `queue_name`.

Subscription patterns (db 0):
- `__keyspace@0__:rq:registry:failed:{queue}` (zadd)
- `__keyspace@0__:rq:registry:canceled:{queue}` (zadd)
- `__keyspace@0__:rq:registry:started:{queue}` (zadd/zrem)

When an event is received:
- failed: page recent IDs via `ZREVRANGEBYSCORE` (bounded LIMIT windows). For each unseen ID: fetch job, summarize, publish `digits.train.failed.v1`, mark in `ProcessedStore`.
- canceled: on `zadd`, publish `digits.train.failed.v1` with `error_kind="user"` and message `"Job canceled"`.
- started: maintain an in-memory map of first-seen start times; on `zrem`, drop the ID. A periodic sweep marks IDs as stuck if older than `stuck_job_timeout_s` and still present (verified via `ZRANK`). Stuck → publish `digits.train.failed.v1` (`error_kind="system"`).

Strict behavior: process exits with a clear error if it cannot subscribe or if `notify-keyspace-events` is not configured as required.

### Ports and typing

- Keep existing `WatcherPorts` style; introduce new typed factories for pubsub and required commands. No `Any`, no casts, no `type: ignore`.
- Local Protocols under `TYPE_CHECKING` for `PubSubProto` and client methods used (`pubsub()`, `config_get`, `psubscribe`, `get_message`, `punsubscribe`, `close`, `zrevrangebyscore`, `zrank`).

### Runner wiring

- Replace `FailureWatcher` with `NotificationWatcher` exclusively. No mode/env toggle. Require envs: `REDIS_URL`, `RQ__QUEUE`, `DIGITS_EVENTS_CHANNEL`, `RQ_WATCHER_STUCK_TIMEOUT_SECONDS`.
- Preflight: assert `notify-keyspace-events` contains `Kz` before entering the main loop; otherwise raise. No fallback behavior.

## Behavior and Semantics

- Avoid full-set scans. Use `ZREVRANGEBYSCORE key +inf -inf LIMIT 0 N` paging until only seen IDs remain. Dedupe persists in `ProcessedStore`.
- PubSub disconnects raise and restart the process via supervisor (no internal fallback loops).

## Command and Cost Comparison (Updated)

Idle:
- Polling: 3 commands/scan × ~45 scans/hour at 2s → ~129,600/day.
- Notify: subscription is one-time; no commands while idle; stuck sweep runs infrequently (configurable), e.g. `ZSCAN/ZRANK` check per candidate only when timeout expires. Typical: O(0–3)/min.

Active (typical training runs):
- Notify issues work only on actual changes (few ZSET pages + job fetches). Even with backlogs, paging is bounded and dedup prevents churn.

Upstash (pay-per-command) ballpark at $0.2/100k commands:
- Idle: polling ≈ $0.26/day vs notify ≈ $0.00–$0.01/day (depending on sweep cadence).
- Busy: notify remains orders of magnitude lower than 2‑second scans.

## Tests and Quality Gates (100% Statement + Branch)

- Unit tests (pure, no Redis):
  - FakePubSub driving `zadd/zrem` sequences; verify publish, dedupe, and stuck detection.
  - Startup behavior: misconfigured `notify-keyspace-events` raises; correct config proceeds.
  - Failure paths: pubsub errors raise; redis command errors bubble and are logged once.
- Runner tests:
  - Required envs enforced; preflight failure raises; single-iteration run loop under monkeypatch for coverage.
- Coverage and lint/type gates remain unchanged and must pass under `make check`.

## Rollout Plan (No Back-Compat)

1. Introduce `NotificationWatcher` and wire runner to it exclusively.
2. Delete polling watcher and its tests in the same PR.
3. Land new unit/integration tests at 100% coverage.
4. Deploy with Redis preconfigured for `notify-keyspace-events=Kz`.

## Configuration Notes (Railway Redis)

- Target: Railway Redis service (managed). Connection via `REDIS_URL` provided by Railway.
- Configure the instance with `notify-keyspace-events=Kz` ahead of deployment. The watcher fails fast if this is not set.
- DB index assumed `0` for keyspace channel prefix (`__keyspace@0__`). If a non‑zero DB is required, adjust the subscription prefix accordingly in the implementation.

## Files to Touch (Strict Plan)

- Add: `src/handwriting_ai/jobs/watcher/notify.py` — event-driven watcher (no Any/casts/ignores).
- Update: `src/handwriting_ai/jobs/watcher/runner.py:1` — wire to `NotificationWatcher` only; enforce preflight.
- Update: `src/handwriting_ai/jobs/watcher/adapters.py:1` — add typed pubsub/client methods.
- Update: `src/handwriting_ai/jobs/watcher/ports.py:1` — add factories for pubsub/config access.
- Add: `tests/test_keyspace_watcher_unit.py`, `tests/test_keyspace_watcher_runner.py` — full coverage.
- Rename: `scripts/rq_failure_watcher.py:1` → `scripts/rq_keyspace_watcher.py` — new entrypoint.
- Replace: `docs/failure_watcher.md:1` → `docs/keyspace_watcher.md` — event-driven only.
- Remove: `src/handwriting_ai/jobs/watcher/watcher.py:1`, `tests/test_failure_watcher*.py`, `scripts/test_keyspace_notifications.py:1` — legacy polling and ad‑hoc demo.

## Appendix — Current Implementation Snapshot

- Polling watcher entry points (to be removed):
  - `src/handwriting_ai/jobs/watcher/watcher.py:21`
  - `src/handwriting_ai/jobs/watcher/runner.py:17`
  - `scripts/rq_failure_watcher.py:1`

This plan retains our strict typing and modular design, removes legacy polling code, and delivers a single, deliberate, event-driven implementation with complete tests and no fallbacks.
