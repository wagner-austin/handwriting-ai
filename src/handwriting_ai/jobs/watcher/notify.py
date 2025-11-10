from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from handwriting_ai.events import digits as ev
from handwriting_ai.jobs.watcher.ports import WatcherPorts, make_default_ports
from handwriting_ai.jobs.watcher.publisher import Publisher, RedisPublisher
from handwriting_ai.jobs.watcher.store import ProcessedStore, RedisProcessedStore


@dataclass
class NotificationWatcher:
    redis_url: str
    queues: tuple[str, ...]
    events_channel: str
    publisher: Publisher | None = None
    store: ProcessedStore | None = None
    ports: WatcherPorts | None = None

    def __post_init__(self) -> None:
        p = self.ports or make_default_ports()
        self.ports = p
        if self.publisher is None:
            self.publisher = RedisPublisher(self.redis_url, redis_factory=p.redis_from_url)
        if self.store is None:
            # Keyed by queue to avoid cross-queue duplication
            # For multi-queue scenarios, include queue in mark calls, not in key
            key = "digits:notify_watcher:processed"
            self.store = RedisProcessedStore(
                self.redis_url, key=key, redis_factory=p.redis_from_url
            )

    def _fail_fast_keyspace(self) -> None:
        assert self.ports is not None
        cfg = self.ports.redis_config_get(self.redis_url, "notify-keyspace-events")
        v = cfg.get("notify-keyspace-events", "")
        # Accept either keyspace (Kz) or keyevent (Ez) families. Require 'z' events.
        has_keyspace = "K" in v and "z" in v
        has_keyevent = "E" in v and "z" in v
        if not (has_keyspace or has_keyevent):
            raise RuntimeError(
                "Redis notify-keyspace-events must include 'Kz' or 'Ez' for zset events"
            )

    def _patterns(self) -> list[str]:
        assert self.ports is not None
        db = self.ports.redis_db_index(self.redis_url)
        prefix = f"__keyspace@{db}__"
        pats: list[str] = []
        regs = ("failed", "scheduled", "canceled", "started")
        if any(q == "*" for q in self.queues):
            for reg in regs:
                pats.append(f"{prefix}:rq:{reg}:*")
            # Also subscribe to keyevent channel for zadd (queue filtering done in handler)
            pats.append(self._keyevent_channel())
            return pats
        for q in self.queues:
            for reg in regs:
                pats.append(f"{prefix}:rq:{reg}:{q}")
        # Always include keyevent channel; handler will filter queues
        pats.append(self._keyevent_channel())
        return pats

    def _keyevent_channel(self) -> str:
        assert self.ports is not None
        db = self.ports.redis_db_index(self.redis_url)
        return f"__keyevent@{db}__:zadd"

    def _queue_accepted(self, queue: str) -> bool:
        if any(q == "*" for q in self.queues):
            return True
        return queue in self.queues

    def _parse_keyspace(self, channel: str, data: str) -> tuple[str, str] | None:
        # Channel format: __keyspace@<db>__:rq:<reg>:<queue>
        if data.lower() != "zadd":
            return None
        parts = channel.split(":")
        # Expect [..., 'rq', reg, queue]
        if len(parts) < 4:
            return None
        if parts[-3] != "rq":
            return None
        reg = parts[-2]
        queue = parts[-1]
        return (reg, queue)

    def _parse_keyevent(self, channel: str, data: str) -> tuple[str, str] | None:
        # Channel format: __keyevent@<db>__:zadd
        # Data is the key: rq:<reg>:<queue>
        if not channel.endswith(":zadd"):
            return None
        parts = data.split(":")
        if len(parts) < 3:
            return None
        if parts[0] != "rq":
            return None
        reg = parts[1]
        queue = parts[2]
        return (reg, queue)

    def _handle_failed(self, queue: str) -> None:
        assert self.ports is not None
        p = self.ports
        conn = p.rq_connect(self.redis_url)
        q = p.rq_queue(conn, queue)
        ids = p.coerce_job_ids(p.rq_failed_registry(q).get_job_ids())
        for jid in ids[:50]:  # bounded page
            key = f"{queue}:{jid}"
            if self.store is not None and self.store.seen(key):
                continue
            try:
                job = p.rq_fetch_job(conn, jid)
            except Exception as e:  # pragma: no cover - bubbled to logs in adapters
                logging.getLogger("handwriting_ai").error(
                    "notify_rq_fetch_failed jid=%s error=%s", jid, e
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
                if hint:  # pragma: no branch - simple truthy check
                    message = p.summarize_exc_info(hint)
            evt = ev.failed(
                ev.Context(
                    request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None
                ),
                error_kind="system",
                message=message,
            )
            self._publish_and_mark(evt, key, reg="failed", queue=queue)

    def _handle_canceled(self, queue: str) -> None:
        assert self.ports is not None
        p = self.ports
        conn = p.rq_connect(self.redis_url)
        q = p.rq_queue(conn, queue)
        ids = p.coerce_job_ids(p.rq_canceled_registry(q).get_job_ids())
        for jid in ids[:50]:
            key = f"{queue}:{jid}"
            if self.store is not None and self.store.seen(key):
                continue
            try:
                job = p.rq_fetch_job(conn, jid)
            except Exception as e:  # pragma: no cover
                logging.getLogger("handwriting_ai").error(
                    "notify_rq_fetch_canceled jid=%s error=%s", jid, e
                )
                raise
            payload = p.extract_payload(job)
            request_id = str(payload.get("request_id") or "")
            user_id_obj: object | None = payload.get("user_id")
            user_id = user_id_obj if isinstance(user_id_obj, int) else 0
            model_id = str(payload.get("model_id") or "")
            evt = ev.failed(
                ev.Context(
                    request_id=request_id, user_id=int(user_id), model_id=model_id, run_id=None
                ),
                error_kind="user",
                message="Job canceled",
            )
            self._publish_and_mark(evt, key, reg="canceled", queue=queue)

    def _publish_and_mark(self, evt: ev.EventV1, jid: str, *, reg: str, queue: str) -> None:
        if self.store is None:  # pragma: no cover - store always set in production
            return
        pub = self.publisher
        try:
            if pub is not None:  # pragma: no branch - publish side-effect
                logging.getLogger("handwriting_ai").info(
                    "notify_publish registry=%s queue=%s", reg, queue
                )
                pub.publish(self.events_channel, ev.encode_event(evt))
        finally:
            self.store.mark(jid)

    def run_forever(self) -> None:  # pragma: no cover - integration path
        self._fail_fast_keyspace()
        assert self.ports is not None
        ps = self.ports.redis_pubsub(self.redis_url)
        try:
            patterns = list(self._patterns())
            for pat in patterns:
                ps.psubscribe(pat)
            log = logging.getLogger("handwriting_ai")
            db = self.ports.redis_db_index(self.redis_url)
            log.info(
                "notify_watcher_started queues=%s db=%s patterns=%s",
                ",".join(self.queues),
                db,
                patterns,
            )
            msg_count = 0
            while True:
                msg = ps.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if not msg:
                    time.sleep(0.1)
                    continue
                msg_count += 1
                log.info(
                    "notify_message_received count=%d channel=%s",
                    msg_count,
                    str(msg.get("channel")),
                )
                self._handle_message(msg)
        finally:
            from contextlib import suppress

            with suppress(Exception):  # pragma: no cover
                ps.close()

    def _handle_message(self, msg: dict[str, object]) -> None:
        """Handle a single PubSub message (pmessage) from keyspace or keyevent."""
        log = logging.getLogger("handwriting_ai")
        ch_obj = msg.get("channel")
        data_obj = msg.get("data")
        ch = ch_obj if isinstance(ch_obj, str) else None
        data = str(data_obj) if data_obj is not None else ""
        if ch is None:
            log.warning("notify_message_skipped reason=no_channel msg=%s", msg)
            return

        # Try keyspace parse first
        parsed = self._parse_keyspace(ch, data)
        if parsed is not None:
            source = "keyspace"
        else:
            # Try keyevent form
            parsed = self._parse_keyevent(ch, data)
            if parsed is None:
                log.info("notify_message_skipped reason=parse_failed channel=%s data=%s", ch, data)
                return
            source = "keyevent"

        reg, queue = parsed
        if not self._queue_accepted(queue):
            return

        logging.getLogger("handwriting_ai").info(
            "notify_event registry=%s queue=%s source=%s", reg, queue, source
        )
        if reg == "failed":
            self._handle_failed(queue)
        elif reg == "canceled":
            self._handle_canceled(queue)
