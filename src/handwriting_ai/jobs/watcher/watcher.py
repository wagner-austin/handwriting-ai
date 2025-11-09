from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from logging import Logger
from typing import TYPE_CHECKING

from handwriting_ai.events import digits as ev

from .ports import WatcherPorts, make_default_ports
from .publisher import Publisher, RedisPublisher
from .store import ProcessedStore, RedisProcessedStore

if TYPE_CHECKING:
    from .adapters import RedisDebugClientProto


@dataclass
class FailureWatcher:
    redis_url: str
    queue_name: str
    events_channel: str
    poll_interval_s: float = 2.0
    stuck_job_timeout_s: float = 1800.0
    publisher: Publisher | None = None
    store: ProcessedStore | None = None
    ports: WatcherPorts | None = None
    ports_provider: Callable[[], WatcherPorts] | None = None

    def __post_init__(self) -> None:
        self.ports_provider = self._make_ports_provider()
        ports = self.ports_provider()
        self.publisher = self._ensure_publisher(ports)
        self.store = self._ensure_store(ports)

    # -- Small helpers to make initialization deterministic for tests -------
    def _make_ports_provider(self) -> Callable[[], WatcherPorts]:
        if self.ports_provider is not None:
            return self.ports_provider
        if self.ports is not None:
            fixed_ports: WatcherPorts = self.ports

            def _prov() -> WatcherPorts:
                return fixed_ports

            return _prov
        return make_default_ports

    def _ensure_publisher(self, ports: WatcherPorts) -> Publisher:
        if self.publisher is not None:
            return self.publisher
        return RedisPublisher(self.redis_url, redis_factory=ports.redis_from_url)

    def _ensure_store(self, ports: WatcherPorts) -> ProcessedStore:
        if self.store is not None:
            return self.store
        key = f"digits:failed_watcher:processed:{self.queue_name}"
        return RedisProcessedStore(self.redis_url, key=key, redis_factory=ports.redis_from_url)

    def scan_once(self) -> None:
        log = logging.getLogger("handwriting_ai")
        log.info("rq_failure_watcher_scan_start queue=%s", self.queue_name)

        assert self.ports_provider is not None
        ports = self.ports_provider()
        conn: RedisDebugClientProto = ports.rq_connect(self.redis_url)
        q = ports.rq_queue(conn, self.queue_name)

        def _process_registry(
            reg_name: str,
            ids: list[str],
            handler: Callable[[RedisDebugClientProto, str], None],
        ) -> None:
            self._log_registry_processing(log, reg_name, ids)
            for jid in ids:
                handler(conn, jid)

        failed_ids = ports.coerce_job_ids(ports.rq_failed_registry(q).get_job_ids())
        _process_registry("failed", failed_ids, self._process_failed_job)

        started_ids = ports.coerce_job_ids(ports.rq_started_registry(q).get_job_ids())
        _process_registry("started", started_ids, self._process_stuck_job)

        stopped_ids = ports.coerce_job_ids(ports.rq_stopped_registry(q).get_job_ids())
        _process_registry("stopped", stopped_ids, self._process_stopped_job)

        canceled_ids = ports.coerce_job_ids(ports.rq_canceled_registry(q).get_job_ids())
        _process_registry("canceled", canceled_ids, self._process_canceled_job)

    def _log_registry_processing(self, log: Logger, reg_name: str, ids: list[str]) -> None:
        if len(ids) > 0:
            log.info(
                "rq_failure_watcher_processing_%s queue=%s job_ids=%s",
                reg_name,
                self.queue_name,
                ids[:10],
            )

    def _process_failed_job(self, conn: RedisDebugClientProto, jid: str) -> None:
        st = self.store
        if st is None:
            return
        if st.seen(jid):
            return
        assert self.ports_provider is not None
        p = self.ports_provider()
        try:
            job = p.rq_fetch_job(conn, jid)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logging.getLogger("handwriting_ai").error(
                "rq_fetch_failed jid=%s error=%s", jid, str(e)
            )
            raise
        payload = p.extract_payload(job)
        request_id = str(payload.get("request_id") or "")
        user_id_obj: object | None = payload.get("user_id")
        user_id = user_id_obj if isinstance(user_id_obj, int) else 0
        model_id = str(payload.get("model_id") or "")
        exc: object = getattr(job, "exc_info", None)
        message = p.summarize_exc_info(exc)
        if not isinstance(exc, str) or message == "job failed":
            hint = p.detect_failed_reason(conn, jid)
            if hint:
                message = p.summarize_exc_info(hint)
        evt = ev.failed(
            ev.Context(request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None),
            error_kind="system",
            message=message,
        )
        self._publish_and_mark(evt, jid)

    def _process_stuck_job(self, conn: RedisDebugClientProto, jid: str) -> None:
        st = self.store
        if st is None:
            return
        if st.seen(jid):
            return
        assert self.ports_provider is not None
        p = self.ports_provider()
        try:
            job = p.rq_fetch_job(conn, jid)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logging.getLogger("handwriting_ai").error(
                "rq_fetch_stuck_job_error jid=%s error=%s", jid, str(e)
            )
            raise
        started_at_obj: object = getattr(job, "started_at", None)
        enqueued_at_obj: object = getattr(job, "enqueued_at", None)
        import datetime as dt

        now = dt.datetime.now(tz=dt.UTC)
        started_at: dt.datetime | None = None
        if isinstance(started_at_obj, dt.datetime):
            started_at = started_at_obj
        elif isinstance(enqueued_at_obj, dt.datetime):
            started_at = enqueued_at_obj
        if started_at is None:
            logging.getLogger("handwriting_ai").debug(
                "rq_stuck_job_no_timestamp jid=%s skipping_timeout_check=true", jid
            )
            return
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=dt.UTC)
        elapsed_s = (now - started_at).total_seconds()
        timeout_s = float(self.stuck_job_timeout_s)
        if elapsed_s < timeout_s:
            return
        payload = p.extract_payload(job)
        request_id = str(payload.get("request_id") or "")
        user_id_obj: object | None = payload.get("user_id")
        user_id = user_id_obj if isinstance(user_id_obj, int) else 0
        model_id = str(payload.get("model_id") or "")
        elapsed_min = int(elapsed_s // 60)
        message = (
            f"Job stuck in started state for {elapsed_min} minutes - "
            "worker likely killed by OS (OOM, signal 9, container eviction). "
            "Reduce batch size or increase memory limit and retry."
        )
        evt = ev.failed(
            ev.Context(request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None),
            error_kind="system",
            message=message,
        )
        self._publish_and_mark(evt, jid)

    def _process_stopped_job(self, conn: RedisDebugClientProto, jid: str) -> None:
        st = self.store
        if st is None:
            return
        if st.seen(jid):
            return
        assert self.ports_provider is not None
        p = self.ports_provider()
        try:
            job = p.rq_fetch_job(conn, jid)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logging.getLogger("handwriting_ai").error(
                "rq_fetch_stopped_job_error jid=%s error=%s", jid, str(e)
            )
            raise
        payload = p.extract_payload(job)
        request_id = str(payload.get("request_id") or "")
        user_id_obj: object | None = payload.get("user_id")
        user_id = user_id_obj if isinstance(user_id_obj, int) else 0
        model_id = str(payload.get("model_id") or "")
        reason = p.detect_failed_reason(conn, jid)
        message = reason if isinstance(reason, str) and reason.strip() else "Job stopped by worker"
        evt = ev.failed(
            ev.Context(request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None),
            error_kind="system",
            message=message,
        )
        self._publish_and_mark(evt, jid)

    def _process_canceled_job(self, conn: RedisDebugClientProto, jid: str) -> None:
        st = self.store
        if st is None:
            return
        if st.seen(jid):
            return
        assert self.ports_provider is not None
        p = self.ports_provider()
        try:
            job = p.rq_fetch_job(conn, jid)
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            logging.getLogger("handwriting_ai").error(
                "rq_fetch_canceled_job_error jid=%s error=%s", jid, str(e)
            )
            raise
        payload = p.extract_payload(job)
        request_id = str(payload.get("request_id") or "")
        user_id_obj: object | None = payload.get("user_id")
        user_id = user_id_obj if isinstance(user_id_obj, int) else 0
        model_id = str(payload.get("model_id") or "")
        message = "Job canceled"
        evt = ev.failed(
            ev.Context(request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None),
            error_kind="user",
            message=message,
        )
        self._publish_and_mark(evt, jid)

    def _publish_and_mark(self, evt: ev.EventV1, jid: str) -> None:
        st = self.store
        if st is None:
            return
        pub = self.publisher
        try:
            if pub is not None:
                pub.publish(self.events_channel, ev.encode_event(evt))
        finally:
            st.mark(jid)

    def run_forever(self) -> None:  # pragma: no cover
        assert self.ports_provider is not None
        log = self.ports_provider().make_logger()
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
