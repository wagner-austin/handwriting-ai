from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import handwriting_ai.jobs.digits as dj
from handwriting_ai.logging import init_logging
from scripts.worker import _make_publisher_from_env, _real_run_training


def _redis_from_url(url: str) -> object:  # pragma: no cover - runtime import only
    import redis

    # RQ stores binary (pickled) payloads; decode_responses must be False
    return redis.from_url(url, decode_responses=False)


@dataclass(frozen=True)
class _RqSettings:
    redis_url: str
    queue_name: str


def _get_env_str(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"{name} is required")
    return v


def _load_rq_settings() -> _RqSettings:
    url = _get_env_str("REDIS_URL")
    q = _get_env_str("RQ__QUEUE")
    return _RqSettings(redis_url=url, queue_name=q)


def _start_worker(s: _RqSettings) -> int:
    # Patch runtime dependencies for the job function
    dj._run_training = _real_run_training
    dj._make_publisher = _make_publisher_from_env

    conn = _redis_from_url(s.redis_url)
    try:
        ok = bool(getattr(conn, "ping", lambda: True)())
    except Exception:  # noqa: BLE001 - connection check only
        ok = False
    if not ok:
        raise RuntimeError("Failed to connect to Redis")

    # Lazily import rq only when actually running the worker, keeping tests light
    import rq as _rq

    q_factory = _rq.Queue
    worker_factory = _rq.Worker
    push_conn = getattr(_rq, "push_connection", None)
    pop_conn = getattr(_rq, "pop_connection", None)
    if callable(push_conn):
        push_conn(conn)
    try:
        q = q_factory(s.queue_name)
        w = worker_factory(queues=[q])
        # Blocks until interrupted; returns boolean success
        success = bool(w.work())
        return 0 if success else 1
    finally:
        if callable(pop_conn):
            pop_conn()


def main(argv: list[str] | None = None) -> int:
    init_logging()
    # argv is unused for now; env controls which queue to listen to
    _ = argv
    settings = _load_rq_settings()
    try:
        return _start_worker(settings)
    except Exception as e:
        logging.getLogger("handwriting_ai").exception("rq_worker_failed: %s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main(sys.argv[1:]))
