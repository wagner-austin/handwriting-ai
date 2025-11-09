from __future__ import annotations

import os

from .watcher import FailureWatcher


def run_from_env() -> None:
    url = os.getenv("REDIS_URL")
    if not url or url.strip() == "":
        raise RuntimeError("REDIS_URL is required")
    q = os.getenv("RQ__QUEUE") or "digits"
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    interval_s = float(os.getenv("RQ_WATCHER_POLL_SECONDS") or 2.0)
    stuck_timeout_s = float(os.getenv("RQ_WATCHER_STUCK_TIMEOUT_SECONDS") or 1800.0)
    FailureWatcher(
        url,
        queue_name=q,
        events_channel=ch,
        poll_interval_s=interval_s,
        stuck_job_timeout_s=stuck_timeout_s,
    ).run_forever()
