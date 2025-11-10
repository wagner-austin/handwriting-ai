from __future__ import annotations

import os

from .notify import NotificationWatcher


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
    queues_raw = os.getenv("RQ__QUEUES", "digits").strip()
    if queues_raw == "*":
        queues: tuple[str, ...] = ("*",)
    else:
        parts: list[str] = [q.strip() for q in queues_raw.split(",") if q.strip()]
        queues = tuple(parts) if len(parts) > 0 else ("digits",)
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    NotificationWatcher(url, queues=queues, events_channel=ch).run_forever()
