from __future__ import annotations

import os

from .watcher import FailureWatcher


def _preflight_registries() -> None:
    """Fail fast if required RQ registries are unavailable.

    This checks that rq.registry exposes StartedJobRegistry and CanceledJobRegistry.
    Avoids long-running loops when the environment provides an incompatible RQ version.
    """
    try:
        import rq  # type: ignore
    except Exception as e:  # pragma: no cover - import error surfaced at runtime
        raise RuntimeError(f"RQ import failed: {type(e).__name__}: {e}")
    reg = getattr(rq, "registry", None)
    if reg is None:
        raise RuntimeError("rq.registry module unavailable")
    missing: list[str] = []
    if not getattr(reg, "StartedJobRegistry", None):
        missing.append("StartedJobRegistry")
    if not getattr(reg, "CanceledJobRegistry", None):
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
