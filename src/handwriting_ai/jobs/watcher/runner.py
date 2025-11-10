from __future__ import annotations

import os

from .notify import NotificationWatcher
from .watcher import FailureWatcher


def _preflight_registries() -> None:
    """Fail fast if required RQ registries are unavailable.

    This checks that rq.registry exposes StartedJobRegistry and CanceledJobRegistry.
    Avoids long-running loops when the environment provides an incompatible RQ version.
    """
    import rq.registry as reg  # will raise ImportError naturally if unavailable

    missing: list[str] = []
    if not hasattr(reg, "StartedJobRegistry"):
        missing.append("StartedJobRegistry")
    if not hasattr(reg, "CanceledJobRegistry"):
        missing.append("CanceledJobRegistry")
    if missing:
        raise RuntimeError(f"Missing required RQ registries: {', '.join(missing)}")


def run_from_env() -> None:
    _preflight_registries()
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


def run_notify_from_env() -> None:
    """Run keyspace-notification based watcher (multi-queue).

    Env vars:
      - REDIS_URL: required
      - RQ__QUEUES: comma-separated queue names (default: "digits")
      - DIGITS_EVENTS_CHANNEL: publish channel (default: "digits:events")
    Fails fast if Redis notify-keyspace-events lacks 'Kz'.
    """
    url = os.getenv("REDIS_URL")
    if not url or url.strip() == "":
        raise RuntimeError("REDIS_URL is required")
    queues_raw = os.getenv("RQ__QUEUES", "digits")
    queues = tuple(q.strip() for q in queues_raw.split(",") if q.strip()) or ("digits",)
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    NotificationWatcher(url, queues=queues, events_channel=ch).run_forever()
