from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
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

    def _rq_connect(url: str) -> object: ...

    def _rq_queue(conn: object, name: str) -> _RQQueueProto: ...

    def _rq_failed_registry(queue: _RQQueueProto) -> _RQRegistryProto: ...

    def _rq_fetch_job(conn: object, job_id: str) -> _RQJobProto: ...

else:  # pragma: no cover - runtime only

    def _rq_connect(url: str):
        import redis

        return redis.Redis.from_url(
            url,
            decode_responses=False,
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
    if isinstance(exc_info, str) and exc_info.strip():
        first = exc_info.strip().splitlines()[0]
        return first[:200]
    return "job failed"


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
        conn = _rq_connect(self.redis_url)
        q = _rq_queue(conn, self.queue_name)
        reg = _rq_failed_registry(q)
        job_ids: list[str] = reg.get_job_ids()
        # Always log scan activity to diagnose silent failures
        log.info(
            "rq_failure_watcher scan queue=%s failed_jobs=%d",
            self.queue_name,
            len(job_ids),
        )
        for jid in job_ids:
            if not isinstance(jid, str):
                continue
            st = self.store
            if st is None:
                continue
            if st.seen(jid):
                continue
            try:
                job = _rq_fetch_job(conn, jid)
            except (RuntimeError, ValueError, TypeError, OSError):
                # If the job cannot be fetched, mark it to avoid loops and continue
                logging.getLogger("handwriting_ai").info("rq_fetch_failed")
                st.mark(jid)
                continue
            payload = _extract_payload(job)
            request_id = str(payload.get("request_id") or "")
            user_id_obj: object | None = payload.get("user_id")
            user_id = user_id_obj if isinstance(user_id_obj, int) else 0
            model_id = str(payload.get("model_id") or "")
            exc: object = getattr(job, "exc_info", None)
            message = _summarize_exc_info(exc)
            # Publish a system failure by default (worker crash or unhandled exception)
            evt = ev.failed(
                ev.Context(
                    request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None
                ),
                error_kind="system",
                message=message,
            )
            pub = self.publisher
            try:
                if pub is not None:
                    log.info(
                        "rq_failure_watcher publish jid=%s req=%s uid=%s model=%s",
                        jid,
                        request_id,
                        int(user_id),
                        model_id,
                    )
                    pub.publish(self.events_channel, ev.encode_event(evt))
            finally:
                log.info("rq_failure_watcher mark_processed jid=%s", jid)
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
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                logging.getLogger("handwriting_ai").info("rq_failure_watcher_scan_error %s", e)
            time.sleep(max(0.1, float(self.poll_interval_s)))


def run_from_env() -> None:
    url = os.getenv("REDIS_URL")
    if not url or url.strip() == "":
        raise RuntimeError("REDIS_URL is required")
    q = os.getenv("RQ__QUEUE") or "digits"
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    interval_s = float(os.getenv("RQ_WATCHER_POLL_SECONDS") or 2.0)
    FailureWatcher(url, queue_name=q, events_channel=ch, poll_interval_s=interval_s).run_forever()
