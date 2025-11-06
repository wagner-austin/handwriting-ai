# RQ Failure Watcher

Publishes `digits.train.failed.v1` for any job that lands in RQ's `FailedJobRegistry`,
covering cases where the worker process crashes or is OOM-killed and cannot emit
its own failure event.

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

- Polls `FailedJobRegistry` for the queue.
- For each new failed job ID, fetches the job and extracts `request_id`, `user_id`, `model_id`
  (when present) from the payload.
- Summarizes `exc_info` first line (or uses `"job failed"`).
- Publishes a `digits.train.failed.v1` event.
- Marks the job ID in a Redis set `digits:failed_watcher:processed:{queue}` to prevent re‑publishing.

## Typing and Dependencies

The watcher uses deferred imports for `redis`/`rq` to keep import‑time dependencies light and
is fully typed without `Any`, `cast`, or `type: ignore`.

