from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import handwriting_ai.jobs.digits as dj
from handwriting_ai.events import digits as _ev
from handwriting_ai.logging import init_logging
from handwriting_ai.monitoring import log_system_info
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
        logging.getLogger("handwriting_ai").info("redis_ping_failed")
        ok = False
    if not ok:
        raise RuntimeError("Failed to connect to Redis")
    # Startup diagnostic to confirm DI wiring and Redis mode
    dec_attr = getattr(conn, "decode_responses", None)
    dec = bool(dec_attr) if isinstance(dec_attr, bool) else False
    logging.getLogger("handwriting_ai").info(
        "digits_wiring_applied queue=%s decode_responses=%s", s.queue_name, bool(dec)
    )

    # Lazily import rq only when actually running the worker, keeping tests light
    import rq as _rq

    q_factory = _rq.Queue
    worker_factory = _rq.Worker

    # Exception handler to emit a failure event for any job error.
    # Signature matches RQ's expected callback: (job, exc_type, exc_value, traceback)
    def _on_rq_job_failure(
        job: object,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        _tb: object | None,
    ) -> None:
        try:
            payload: dict[str, object] | None = None
            args_obj = getattr(job, "args", None)
            if (
                isinstance(args_obj, list | tuple)
                and len(args_obj) > 0
                and isinstance(args_obj[0], dict)
            ):
                payload = args_obj[0]
            request_id = str(payload.get("request_id") if payload else "")
            user_id_obj = payload.get("user_id") if payload else None
            user_id = int(user_id_obj) if isinstance(user_id_obj, int) else 0
            model_id = str(payload.get("model_id") if payload else "")
            err_name = exc_type.__name__ if exc_type is not None else "Error"
            err_msg = str(exc_value) if exc_value is not None else "job failed"
            msg = f"{err_name}: {err_msg}" if err_msg else err_name
            pub = _make_publisher_from_env()
            ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
            if pub is not None:
                evt = _ev.failed(
                    _ev.Context(
                        request_id=request_id,
                        user_id=int(user_id),
                        model_id=model_id,
                        run_id=None,
                    ),
                    error_kind="system",
                    message=msg,
                )
                pub.publish(ch, _ev.encode_event(evt))
        except (OSError, RuntimeError, ValueError, TypeError):
            # Never let the exception handler crash the worker loop
            logging.getLogger("handwriting_ai").info("rq_worker_emit_failed_event_error")

    push_conn = getattr(_rq, "push_connection", None)
    pop_conn = getattr(_rq, "pop_connection", None)
    if callable(push_conn):
        push_conn(conn)
    try:
        q = q_factory(s.queue_name)
        # Strict: require modern RQ that accepts exception_handlers; no legacy fallbacks
        w = worker_factory(queues=[q], exception_handlers=[_on_rq_job_failure])
        # Blocks until interrupted; returns boolean success
        success = bool(w.work())
        return 0 if success else 1
    finally:
        if callable(pop_conn):
            pop_conn()


def main(argv: list[str] | None = None) -> int:
    init_logging()
    # Log basic system info to correlate with runtime behavior
    log_system_info()
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
