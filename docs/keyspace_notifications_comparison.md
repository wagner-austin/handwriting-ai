# Redis Keyspace Notifications — Completed Migration (No Fallback)

This document confirms our completed migration from polling to Redis keyspace notifications for RQ registries. The design is strict: no fallbacks, no back-compat toggles, and legacy polling code removed. All changes preserve our strict typing (no Any/casts/ignores), DRY structure, and 100% statement/branch coverage under `make check`.

## Repository State

- Event-driven watcher: `src/handwriting_ai/jobs/watcher/notify.py` implements `NotificationWatcher`, subscribing to Redis keyspace notifications for RQ registries.
- Integration boundaries are abstracted through `WatcherPorts` for redis/rq access: `src/handwriting_ai/jobs/watcher/ports.py:1` and `src/handwriting_ai/jobs/watcher/adapters.py:1`.
- Failure heuristics and payload extraction are typed helpers: `src/handwriting_ai/jobs/watcher/logic.py:1`.
- Publisher and dedupe store are small, typed components: `src/handwriting_ai/jobs/watcher/publisher.py:1`, `src/handwriting_ai/jobs/watcher/store.py:1`.
- Entry point: `scripts/rq_keyspace_watcher.py` invokes `run_notify_from_env()`.
- Runner: `src/handwriting_ai/jobs/watcher/runner.py` exposes only `run_notify_from_env()`.
- Integration boundaries remain abstracted through `WatcherPorts` and `adapters`, with added pubsub/config helpers.
- Typed helpers retained: `logic.py` (payload extraction, reason detection), `publisher.py`, `store.py`.

Conclusion: We run event-driven only; polling code and tests have been removed.

## Why Keyspace Notifications

- Idle cost and noise: polling touches three registries per scan (failed/started/canceled), even when nothing changes.
- Event-driven flips the model: subscribe once; react only when Redis reports a ZSET change on the specific registry keys.

## Target Design (Event-Driven Only)

Synchronous watcher consumes keyspace notifications for the RQ registries: `failed`, `scheduled` (optional), `started`, and `canceled` for configured queues.

Subscription patterns (db 0):
- `__keyspace@0__:rq:registry:failed:{queue}` (zadd)
- `__keyspace@0__:rq:registry:canceled:{queue}` (zadd)
- `__keyspace@0__:rq:registry:started:{queue}` (zadd/zrem)

When an event is received:
- failed: page recent IDs via bounded queries. For each unseen ID: fetch job, summarize, publish `digits.train.failed.v1`, mark in `ProcessedStore`.
- canceled: on `zadd`, publish `digits.train.failed.v1` with `error_kind="user"` and message `"Job canceled"`.
- started/scheduled: optional future enhancements.

Strict behavior: process exits with a clear error if it cannot subscribe or if `notify-keyspace-events` is not configured as required.

### Ports and typing

- Keep existing `WatcherPorts` style; introduce new typed factories for pubsub and required commands. No `Any`, no casts, no `type: ignore`.
- Local Protocols under `TYPE_CHECKING` for `PubSubProto` and client methods used (`pubsub()`, `config_get`, `psubscribe`, `get_message`, `punsubscribe`, `close`, `zrevrangebyscore`, `zrank`).

### Runner wiring

- Use `NotificationWatcher` exclusively. No mode/env toggle. Require envs: `REDIS_URL`, `RQ__QUEUES`, `DIGITS_EVENTS_CHANNEL`.
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

## Operational Notes

- Configure Redis with `notify-keyspace-events=Kz`.
- Set env for the process:
  - `REDIS_URL` (required)
  - `RQ__QUEUES` (comma-separated queue names; default: `digits`)
    - Use `*` to watch all queues dynamically without enumerating them
  - `DIGITS_EVENTS_CHANNEL` (default: `digits:events`)
- Start the watcher: `python scripts/rq_keyspace_watcher.py`.

## Configuration Notes (Railway Redis)

- Target: Railway Redis service (managed). Connection via `REDIS_URL` provided by Railway.
- Configure the instance with `notify-keyspace-events=Kz` ahead of deployment. The watcher fails fast if this is not set.
- DB index assumed `0` for keyspace channel prefix (`__keyspace@0__`). If a non‑zero DB is required, adjust the subscription prefix accordingly in the implementation.

## Implementation Status

- Implemented: NotificationWatcher, ports/adapters extensions, runner entrypoint, and unit tests.
- Removed: legacy polling watcher, tests, and old script.

## Appendix — Current Implementation Snapshot

- Polling watcher entry points (to be removed):
  - `src/handwriting_ai/jobs/watcher/watcher.py:21`
  - `src/handwriting_ai/jobs/watcher/runner.py:17`
  - `scripts/rq_failure_watcher.py:1`

This plan retains our strict typing and modular design, removes legacy polling code, and delivers a single, deliberate, event-driven implementation with complete tests and no fallbacks.
