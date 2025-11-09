# RQ Failure Watcher

Publishes `digits.train.failed.v1` for any job that lands in RQ's failure-related registries,
covering cases where the worker process crashes, is OOM-killed, is stopped, or is canceled and
cannot emit its own failure event.

## Running

Run as a separate sidecar process alongside the worker:

```
python scripts/rq_failure_watcher.py
```

Environment variables:

- `REDIS_URL` (required): Redis connection URL
- `RQ__QUEUE` (required): Queue name to watch (e.g. `digits`)
- `DIGITS_EVENTS_CHANNEL` (optional, default `digits:events`): PubSub channel for events
- `RQ_WATCHER_POLL_SECONDS` (optional, default `2.0`): Poll interval seconds

## Behavior

- Polls `FailedJobRegistry` for explicit failures.
- Polls `StartedJobRegistry` for stuck jobs that exceed a timeout (e.g., worker OOM-killed).
- Polls `StoppedJobRegistry` for jobs stopped by the worker.
- Polls `CanceledJobRegistry` for user/system canceled jobs and emits a user-kind failed event.
- For each new failed job ID, fetches the job and extracts `request_id`, `user_id`, `model_id`
  (when present) from the payload.
- Summarizes `exc_info` first line (or uses `"job failed"`).
- Publishes a `digits.train.failed.v1` event.
- Marks the job ID in a Redis set `digits:failed_watcher:processed:{queue}` to prevent re‑publishing.

## API and Semantics

- Import from `handwriting_ai.jobs.watcher` (compatibility facade removed).
- The watcher raises exceptions on failures in publisher/store/adapters and in per-job handlers; the
  outer `run_forever` loop logs errors and continues scanning.

## Typing and Dependencies

The watcher uses deferred imports for `redis`/`rq` to keep import-time dependencies light and is fully
typed without `Any`, `cast`, or `type: ignore`.
