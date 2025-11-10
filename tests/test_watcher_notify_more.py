from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

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


def _ports_stub(
    *, failed: list[str], started: list[str], canceled: list[str], db: int = 0, cfg_val: str = "Kz"
) -> WatcherPorts:
    return WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=lambda u: _dummy_conn(),
        rq_queue=lambda c, n: object(),
        rq_failed_registry=lambda _q: _Reg(failed),
        rq_started_registry=lambda _q: _Reg(started),
        rq_canceled_registry=lambda _q: _Reg(canceled),
        rq_fetch_job=lambda _c, _j: _Job(
            {"request_id": "r1", "user_id": 1, "model_id": "m"}, "trace"
        ),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
        redis_pubsub=lambda url: _PS(),
        redis_config_get=lambda url, p: {"notify-keyspace-events": cfg_val},
        redis_db_index=lambda url: db,
    )


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


class _PS:
    def psubscribe(self, *patterns: str) -> None:  # pragma: no cover - not exercised here
        return None

    def get_message(
        self, ignore_subscribe_messages: bool = True, timeout: float | None = None
    ) -> dict[str, object] | None:
        return None

    def close(self) -> None:  # pragma: no cover - not exercised here
        return None


def test_fail_fast_keyspace_raises_when_missing() -> None:
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        ports=_ports_stub(failed=[], started=[], canceled=[], cfg_val=""),
    )
    with pytest.raises(RuntimeError):
        w._fail_fast_keyspace()


def test_patterns_multi_queue() -> None:
    w = NotificationWatcher(
        redis_url="redis://localhost/2",
        queues=("a", "b"),
        events_channel="digits:events",
        ports=_ports_stub(failed=[], started=[], canceled=[], db=2),
    )
    pats = w._patterns()
    assert "__keyspace@2__:rq:registry:failed:a" in pats
    assert "__keyspace@2__:rq:registry:started:b" in pats


def test_patterns_wildcard_all_queues() -> None:
    w = NotificationWatcher(
        redis_url="redis://localhost/5",
        queues=("*",),
        events_channel="digits:events",
        ports=_ports_stub(failed=[], started=[], canceled=[], db=5),
    )
    pats = w._patterns()
    assert "__keyspace@5__:rq:registry:failed:*" in pats
    assert "__keyspace@5__:rq:registry:started:*" in pats


def test_handle_message_non_zadd_ignored() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-2"], started=[], canceled=[]),
    )
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyspace@0__:rq:registry:failed:digits",
        "channel": "__keyspace@0__:rq:registry:failed:digits",
        "data": "zrem",
    }
    w._handle_message(msg)
    assert st.seen_ids == set() and pub.items == []


def test_handle_message_short_channel_ignored() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-3"], started=[], canceled=[]),
    )
    msg: dict[str, object] = {"type": "pmessage", "channel": "foo", "data": "zadd"}
    w._handle_message(msg)
    assert st.seen_ids == set() and pub.items == []


def test_canceled_event_path_marks_and_publishes() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=[], started=[], canceled=["jid-4"]),
    )
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyspace@0__:rq:registry:canceled:digits",
        "channel": "__keyspace@0__:rq:registry:canceled:digits",
        "data": "zadd",
    }
    w._handle_message(msg)
    assert "digits:jid-4" in st.seen_ids and len(pub.items) == 1
