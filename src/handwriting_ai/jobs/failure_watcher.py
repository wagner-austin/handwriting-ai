from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, runtime_checkable
from typing import Protocol as _Protocol

from handwriting_ai.events import digits as ev

if TYPE_CHECKING:

    class _RedisClientProto(_Protocol):  # pragma: no cover - typing only
        def publish(self, channel: str, message: str) -> int: ...
        def sismember(self, key: str, member: str) -> bool: ...
        def sadd(self, key: str, *members: str) -> int: ...

    def _redis_from_url(url: str, *, decode_responses: bool = False) -> _RedisClientProto: ...

else:  # pragma: no cover - runtime only

    def _redis_from_url(url: str, *, decode_responses: bool = False):
        import redis

        return redis.Redis.from_url(url, decode_responses=decode_responses)


def _make_logger() -> logging.Logger:
    return logging.getLogger("handwriting_ai")


class Publisher(_Protocol):  # narrow protocol; avoids heavy jobs import
    def publish(self, channel: str, message: str) -> int: ...


class _ProcessedStore(_Protocol):  # pragma: no cover - typing only
    def seen(self, job_id: str) -> bool: ...

    def mark(self, job_id: str) -> None: ...


class _RedisPublisher:
    def __init__(self, url: str) -> None:
        self._url = url

    def publish(self, channel: str, message: str) -> int:
        try:
            client = _redis_from_url(self._url)
            out_val: int = int(client.publish(channel, message))
            return out_val
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError):
            logging.getLogger("handwriting_ai").info("redis_publish_failed")
            return 0


class _RedisProcessedStore:
    def __init__(self, url: str, *, key: str) -> None:
        self._url = url
        self._key = key

    def seen(self, job_id: str) -> bool:
        try:
            client = _redis_from_url(self._url)
            val: bool = bool(client.sismember(self._key, job_id))
            return val
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError):
            logging.getLogger("handwriting_ai").info("redis_seen_failed")
            return False

    def mark(self, job_id: str) -> None:
        try:
            client = _redis_from_url(self._url)
            client.sadd(self._key, job_id)
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError):
            logging.getLogger("handwriting_ai").info("redis_mark_failed")


if TYPE_CHECKING:

    class _RQQueueProto(_Protocol):  # pragma: no cover - typing only
        ...

    class _RQRegistryProto(_Protocol):  # pragma: no cover - typing only
        def get_job_ids(self) -> list[str]: ...

    class _RQJobProto(_Protocol):  # pragma: no cover - typing only
        args: object
        exc_info: object

    class _RedisDebugClientProto(_Protocol):  # pragma: no cover - typing only
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]: ...

    def _rq_connect(url: str) -> _RedisDebugClientProto: ...

    def _rq_queue(conn: _RedisDebugClientProto, name: str) -> _RQQueueProto: ...

    def _rq_failed_registry(queue: _RQQueueProto) -> _RQRegistryProto: ...

    def _rq_fetch_job(conn: _RedisDebugClientProto, job_id: str) -> _RQJobProto: ...

else:  # pragma: no cover - runtime only

    def _rq_connect(url: str):
        import redis

        return redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=10.0,
        )

    def _rq_queue(conn, name):
        import rq

        return rq.Queue(name, connection=conn)

    def _rq_failed_registry(queue):
        import rq

        return rq.registry.FailedJobRegistry(queue=queue)

    def _rq_fetch_job(conn, job_id):
        import rq

        return rq.job.Job.fetch(job_id, connection=conn)


def _summarize_exc_info(exc_info: object) -> str:
    """Extract meaningful error from exc_info, detecting OOM kills."""
    if isinstance(exc_info, str) and exc_info.strip():
        lines = exc_info.strip().splitlines()
        # Check for OOM/signal 9 indicators
        full_text = "\n".join(lines[:10])  # Check first 10 lines
        if "signal 9" in full_text.lower() or "sigkill" in full_text.lower():
            return (
                "OOM kill detected (signal 9 / SIGKILL) - "
                "worker terminated by system due to memory exhaustion"
            )
        if "waitpid returned 9" in full_text:
            return "Worker killed by OS (signal 9) - likely OOM (out of memory)"
        # Return first line with more context
        return lines[0][:300] if lines else "job failed"
    return "job failed"


def _coerce_str(val: object) -> str | None:
    if isinstance(val, str):
        return val
    if isinstance(val, bytes | bytearray):
        try:
            return bytes(val).decode("utf-8", errors="ignore")
        except (UnicodeDecodeError, ValueError):  # pragma: no cover - defensive
            logging.getLogger("handwriting_ai").debug("coerce_str_decode_failed")
            return None
    return None


def _extract_payload(job: object) -> dict[str, object]:
    # RQ job.args is typically a tuple[list] of positional args
    args: object = getattr(job, "args", None)
    if isinstance(args, list | tuple) and len(args) > 0:
        first = args[0]
        if isinstance(first, dict):
            out: dict[str, object] = {}
            for k in ("request_id", "user_id", "model_id", "type"):
                out[k] = first.get(k)
            return out
    return {}


@dataclass
class FailureWatcher:
    redis_url: str
    queue_name: str
    events_channel: str
    poll_interval_s: float = 2.0
    publisher: Publisher | None = None
    store: _ProcessedStore | None = None

    def __post_init__(self) -> None:
        if self.publisher is None:
            self.publisher = _RedisPublisher(self.redis_url)
        if self.store is None:
            key = f"digits:failed_watcher:processed:{self.queue_name}"
            self.store = _RedisProcessedStore(self.redis_url, key=key)

    def scan_once(self) -> None:
        log = logging.getLogger("handwriting_ai")
        log.info("rq_failure_watcher_scan_start queue=%s", self.queue_name)

        conn = _rq_connect(self.redis_url)
        log.debug("rq_failure_watcher_connected redis_url=%s", self.redis_url)

        q = _rq_queue(conn, self.queue_name)
        log.debug("rq_failure_watcher_queue_created queue=%s", self.queue_name)

        reg = _rq_failed_registry(q)
        log.debug("rq_failure_watcher_registry_created queue=%s", self.queue_name)

        # Get job IDs from RQ FailedJobRegistry
        raw_job_ids = reg.get_job_ids()
        log.info(
            "rq_failure_watcher_registry_fetched queue=%s raw_type=%s raw_count=%d",
            self.queue_name,
            type(raw_job_ids).__name__,
            len(raw_job_ids) if isinstance(raw_job_ids, list) else 0,
        )

        job_ids = _coerce_job_ids(raw_job_ids)
        log.info(
            "rq_failure_watcher_scan_complete queue=%s failed_jobs=%d",
            self.queue_name,
            len(job_ids),
        )

        if len(job_ids) > 0:
            log.info(
                "rq_failure_watcher_processing queue=%s job_ids=%s",
                self.queue_name,
                job_ids[:10],
            )

        for jid in job_ids:
            if not isinstance(jid, str):
                log.warning("rq_failure_watcher_skip_non_string jid_type=%s", type(jid).__name__)
                continue
            self._process_failed_job(conn, jid)

    def _process_failed_job(self, conn: _RedisDebugClientProto, jid: str) -> None:
        st = self.store
        if st is None:
            return
        if st.seen(jid):
            return
        try:
            job = _rq_fetch_job(conn, jid)
        except (RuntimeError, ValueError, TypeError, OSError):
            logging.getLogger("handwriting_ai").info("rq_fetch_failed")
            st.mark(jid)
            return
        payload = _extract_payload(job)
        request_id = str(payload.get("request_id") or "")
        user_id_obj: object | None = payload.get("user_id")
        user_id = user_id_obj if isinstance(user_id_obj, int) else 0
        model_id = str(payload.get("model_id") or "")
        exc: object = getattr(job, "exc_info", None)
        message = _summarize_exc_info(exc)
        if not isinstance(exc, str) or message == "job failed":
            hint = _detect_failed_reason(conn, jid)
            if hint:
                message = _summarize_exc_info(hint)

        log = logging.getLogger("handwriting_ai")
        log.info(
            "rq_failure_detected jid=%s req=%s uid=%s model=%s error=%s",
            jid,
            request_id,
            int(user_id),
            model_id,
            message[:200],
        )
        if exc and isinstance(exc, str):
            log.debug("rq_failure_exc_info jid=%s exc=%s", jid, exc[:500])

        evt = ev.failed(
            ev.Context(request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None),
            error_kind="system",
            message=message,
        )
        pub = self.publisher
        try:
            if pub is not None:
                log.info(
                    "rq_failure_watcher_publish jid=%s req=%s uid=%s model=%s channel=%s",
                    jid,
                    request_id,
                    int(user_id),
                    model_id,
                    self.events_channel,
                )
                pub.publish(self.events_channel, ev.encode_event(evt))
            else:
                log.warning("rq_failure_watcher_no_publisher jid=%s", jid)
        finally:
            log.info("rq_failure_watcher_mark_processed jid=%s", jid)
            st.mark(jid)

    def run_forever(self) -> None:  # pragma: no cover - loop integration tested via scan_once
        log = _make_logger()
        log.info(
            "rq_failure_watcher starting queue=%s channel=%s interval=%.2fs",
            self.queue_name,
            self.events_channel,
            float(self.poll_interval_s),
        )
        while True:
            try:
                self.scan_once()
            except Exception as e:
                logging.getLogger("handwriting_ai").error(
                    "rq_failure_watcher_scan_error type=%s error=%s",
                    type(e).__name__,
                    str(e),
                    exc_info=True,
                )
            time.sleep(max(0.1, float(self.poll_interval_s)))


def _coerce_job_ids(items: Sequence[object]) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, bytes):
            try:
                out.append(it.decode("utf-8"))
            except UnicodeDecodeError as exc:
                logging.getLogger("handwriting_ai").debug("jid_decode_failed %s", exc)
                continue
    return out


def run_from_env() -> None:
    url = os.getenv("REDIS_URL")
    if not url or url.strip() == "":
        raise RuntimeError("REDIS_URL is required")
    q = os.getenv("RQ__QUEUE") or "digits"
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    interval_s = float(os.getenv("RQ_WATCHER_POLL_SECONDS") or 2.0)
    FailureWatcher(url, queue_name=q, events_channel=ch, poll_interval_s=interval_s).run_forever()


@runtime_checkable
class _RedisHashProto(_Protocol):  # pragma: no cover - typing only
    def hget(self, key: str, field: str) -> str | bytes | None: ...


def _detect_failed_reason(conn: object, job_id: str) -> str | None:
    """Best-effort fetch of failure reason from Redis job hash fields.

    Attempts common fields like 'failed_reason', 'failure_reason', and 'exc_info'.
    Returns a string if found; otherwise None. Safe no-op on unsupported clients.
    """
    if not isinstance(conn, _RedisHashProto):
        return None
    key = f"rq:job:{job_id}"
    for field in ("failed_reason", "failure_reason", "exc_info"):
        try:
            raw = conn.hget(key, field)
        except (RuntimeError, ValueError, TypeError, OSError) as _e:
            logging.getLogger("handwriting_ai").debug("redis_hget_failed %s", _e)
            continue
        s = _coerce_str(raw)
        if s and s.strip():
            return s
    return None
