from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.jobs.watcher.notify import NotificationWatcher
from handwriting_ai.jobs.watcher.ports import WatcherPorts

if TYPE_CHECKING:
    from handwriting_ai.jobs.watcher.adapters import (
        RedisClient as _RedisClient,
    )
    from handwriting_ai.jobs.watcher.adapters import (
        RedisDebugClientProto as _RedisDebugClientProto,
    )


@dataclass
class _Pub:
    items: list[tuple[str, str]] = field(default_factory=list)

    def publish(self, channel: str, message: str) -> int:
        self.items.append((channel, message))
        return 1


@dataclass
class _Store:
    seen_ids: set[str] = field(default_factory=set)

    def seen(self, job_id: str) -> bool:
        return job_id in self.seen_ids

    def mark(self, job_id: str) -> None:
        self.seen_ids.add(job_id)


def _rfu(url: str, *, decode_responses: bool = False) -> _RedisClient:  # pragma: no cover - minimal
    class _C:
        def publish(self, channel: str, message: str) -> int:
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return False

        def sadd(self, key: str, *members: str) -> int:
            return len(members)

    return _C()


def _dummy_conn() -> _RedisDebugClientProto:  # pragma: no cover - minimal
    class _C:
        def zrange(self, key: str, start: int, end: int) -> list[str]:
            return []

        def hget(self, key: str, field: str) -> str | None:
            return None

    return _C()


class _Reg:
    def __init__(self, ids: list[str]):
        self._ids = ids

    def get_job_ids(self) -> list[str]:
        return list(self._ids)


@dataclass
class _Job:
    args: object
    exc_info: object
    started_at: object
    enqueued_at: object

    def __init__(self, payload: object, exc_info: str | None) -> None:
        self.args = [payload]
        self.exc_info = exc_info
        self.started_at = None
        self.enqueued_at = None


def _ports_with_failed_ids(ids: list[str]) -> WatcherPorts:
    return WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=lambda u: _dummy_conn(),
        rq_queue=lambda c, n: object(),
        rq_failed_registry=lambda _q: _Reg(ids),
        rq_started_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=lambda _c, _j: _Job(
            {"request_id": "r1", "user_id": 1, "model_id": "m"}, "trace"
        ),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )


def test_notify_watcher_handles_failed_event() -> None:
    # Simulate a failed zadd event for digits queue
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_with_failed_ids(["jid-1"]),
    )
    # Message structure from redis-py pubsub
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyspace@0__:rq:failed:digits",
        "channel": "__keyspace@0__:rq:failed:digits",
        "data": "zadd",
    }
    w._handle_message(msg)
    assert "digits:jid-1" in st.seen_ids and len(pub.items) == 1
