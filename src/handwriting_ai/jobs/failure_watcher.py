from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

from handwriting_ai.events import digits as ev
from typing import Protocol as _Protocol


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
        try:  # defer import for tests and to avoid hard runtime dep if not used
            import redis

            client = redis.Redis.from_url(self._url)
            return int(client.publish(channel, message))
        except Exception:
            _make_logger().info("redis_publish_failed")
            return 0


class _RedisProcessedStore:
    def __init__(self, url: str, *, key: str) -> None:
        self._url = url
        self._key = key

    def seen(self, job_id: str) -> bool:
        try:
            import redis

            client = redis.Redis.from_url(self._url)
            val = client.sismember(self._key, job_id)
            return bool(val)
        except Exception:
            _make_logger().info("redis_seen_failed")
            return False

    def mark(self, job_id: str) -> None:
        try:
            import redis

            client = redis.Redis.from_url(self._url)
            client.sadd(self._key, job_id)
        except Exception:
            _make_logger().info("redis_mark_failed")


def _rq_connect(url: str) -> object:
    try:
        import redis

        return redis.Redis.from_url(url, decode_responses=False)
    except Exception as e:  # pragma: no cover - connectivity/environment
        raise RuntimeError("failed to connect to redis") from e


def _rq_queue(conn: object, name: str) -> object:
    import rq

    return rq.Queue(name, connection=conn)


def _rq_failed_registry(queue: object) -> object:
    import rq

    return rq.registry.FailedJobRegistry(queue=queue)


def _rq_fetch_job(conn: object, job_id: str) -> object:
    import rq

    return rq.job.Job.fetch(job_id, connection=conn)


def _summarize_exc_info(exc_info: object) -> str:
    if isinstance(exc_info, str) and exc_info.strip():
        first = exc_info.strip().splitlines()[0]
        return first[:200]
    return "job failed"


def _extract_payload(job: object) -> dict[str, object]:
    # RQ job.args is typically a tuple[list] of positional args
    args = getattr(job, "args", None)
    if isinstance(args, (list, tuple)) and args:
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
        conn = _rq_connect(self.redis_url)
        q = _rq_queue(conn, self.queue_name)
        reg = _rq_failed_registry(q)
        job_ids = list(getattr(reg, "get_job_ids", lambda: [])())
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
            except Exception:
                # If the job cannot be fetched, mark it to avoid loops and continue
                st.mark(jid)
                continue
            payload = _extract_payload(job)
            request_id = str(payload.get("request_id") or "")
            try:
                user_id_obj = payload.get("user_id")
                user_id = int(user_id_obj) if not isinstance(user_id_obj, bool) else 0
            except Exception:
                user_id = 0
            model_id = str(payload.get("model_id") or "")
            message = _summarize_exc_info(getattr(job, "exc_info", None))
            # Publish a system failure by default (worker crash or unhandled exception)
            evt = ev.failed(
                ev.Context(request_id=request_id, user_id=user_id, model_id=model_id, run_id=None),
                error_kind="system",
                message=message,
            )
            pub = self.publisher
            try:
                if pub is not None:
                    pub.publish(self.events_channel, ev.encode_event(evt))
            finally:
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
                log.info("rq_failure_watcher_scan_error %s", e)
            time.sleep(max(0.1, float(self.poll_interval_s)))


def run_from_env() -> None:
    url = os.getenv("REDIS_URL")
    if not url or url.strip() == "":
        raise RuntimeError("REDIS_URL is required")
    q = os.getenv("RQ__QUEUE") or "digits"
    ch = os.getenv("DIGITS_EVENTS_CHANNEL") or "digits:events"
    interval_s = float(os.getenv("RQ_WATCHER_POLL_SECONDS") or 2.0)
    FailureWatcher(url, queue_name=q, events_channel=ch, poll_interval_s=interval_s).run_forever()
