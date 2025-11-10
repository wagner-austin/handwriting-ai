from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.events import digits as ev
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
    assert "__keyevent@2__:zadd" in pats


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
    assert "__keyevent@5__:zadd" in pats


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


def test_keyevent_failed_path_marks_and_publishes() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-5"], started=[], canceled=[]),
    )
    # Keyevent: channel is __keyevent@db__:zadd; data is key name
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyevent@0__:zadd",
        "channel": "__keyevent@0__:zadd",
        "data": "rq:registry:failed:digits",
    }
    w._handle_message(msg)
    assert "digits:jid-5" in st.seen_ids and len(pub.items) == 1


def test_keyevent_queue_filtering_explicit() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-6"], started=[], canceled=[]),
    )
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyevent@0__:zadd",
        "channel": "__keyevent@0__:zadd",
        "data": "rq:registry:failed:other",
    }
    w._handle_message(msg)
    assert st.seen_ids == set() and pub.items == []


def test_keyevent_wildcard_accepts_any_queue() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("*",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-7"], started=[], canceled=[]),
    )
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyevent@0__:zadd",
        "channel": "__keyevent@0__:zadd",
        "data": "rq:registry:failed:anyqueue",
    }
    w._handle_message(msg)
    assert any(it[0] == "digits:events" for it in pub.items)


def test_fail_fast_accepts_keyevent_only() -> None:
    # Only Ez provided; should not raise
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        ports=_ports_stub(failed=[], started=[], canceled=[], cfg_val="Ez"),
    )
    # No exception expected
    w._fail_fast_keyspace()


def test_parse_keyspace_wrong_prefix_and_keyevent_non_key() -> None:
    # Wrong keyspace prefix (not rq:registry) -> ignored; also not a keyevent form
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-x"], started=[], canceled=[]),
    )
    msg: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyspace@0__:foo:registry:failed:digits",
        "channel": "__keyspace@0__:foo:registry:failed:digits",
        "data": "zadd",
    }
    w._handle_message(msg)
    assert st.seen_ids == set() and pub.items == []


def test_parse_keyevent_bad_data_and_wrong_prefix() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-y"], started=[], canceled=[]),
    )
    # Too-short data
    msg1: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyevent@0__:zadd",
        "channel": "__keyevent@0__:zadd",
        "data": "rq:registry",
    }
    w._handle_message(msg1)
    # Wrong prefix
    msg2: dict[str, object] = {
        "type": "pmessage",
        "pattern": "__keyevent@0__:zadd",
        "channel": "__keyevent@0__:zadd",
        "data": "foo:bar:failed:digits",
    }
    w._handle_message(msg2)
    assert st.seen_ids == set() and pub.items == []


def test_handle_message_channel_none_ignored() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=[], started=[], canceled=[]),
    )
    msg: dict[str, object] = {"type": "pmessage", "channel": None, "data": "zadd"}
    w._handle_message(msg)
    assert st.seen_ids == set() and pub.items == []


def test_publish_and_mark_handles_store_none_and_pub_none() -> None:
    # store None -> early return, no error
    w1 = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=None,
        store=None,
        ports=_ports_stub(failed=[], started=[], canceled=[]),
    )
    evt = ev.failed(
        ev.Context(request_id="r", user_id=1, model_id="m", run_id=None),
        error_kind="system",
        message="x",
    )
    # Should not raise
    w1._publish_and_mark(evt, "jid", reg="failed", queue="digits")

    # pub None but store present -> mark only
    st = _Store()
    w2 = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=None,
        store=st,
        ports=_ports_stub(failed=[], started=[], canceled=[]),
    )
    w2._publish_and_mark(evt, "digits:x", reg="failed", queue="digits")
    assert "digits:x" in st.seen_ids


def test_failed_and_canceled_seen_skips_publish() -> None:
    pub = _Pub()
    st = _Store(seen_ids={"digits:jid-dup", "digits:jid-dup2"})
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=["jid-dup"], started=[], canceled=["jid-dup2"]),
    )
    w._handle_message(
        {
            "type": "pmessage",
            "pattern": "__keyspace@0__:rq:failed:digits",
            "channel": "__keyspace@0__:rq:failed:digits",
            "data": "zadd",
        }
    )
    w._handle_message(
        {
            "type": "pmessage",
            "pattern": "__keyspace@0__:rq:canceled:digits",
            "channel": "__keyspace@0__:rq:canceled:digits",
            "data": "zadd",
        }
    )
    # No new publishes
    assert pub.items == []


def test_failed_message_uses_detect_failed_reason_hint() -> None:
    pub = _Pub()
    st = _Store()

    def _ports_with_hint() -> WatcherPorts:
        p = _ports_stub(failed=["jid-hint"], started=[], canceled=[])

        def _detected(_conn: object, _jid: str) -> str | None:
            return "HINT"

        def _sum(val: object) -> str:
            return f"SUM:{val}"

        def _rfj(_c: object, _j: str) -> _Job:
            return _Job({"request_id": "r1", "user_id": 1, "model_id": "m"}, None)

        return WatcherPorts(
            redis_from_url=p.redis_from_url,
            rq_connect=p.rq_connect,
            rq_queue=p.rq_queue,
            rq_failed_registry=p.rq_failed_registry,
            rq_started_registry=p.rq_started_registry,
            rq_canceled_registry=p.rq_canceled_registry,
            rq_fetch_job=_rfj,
            coerce_job_ids=p.coerce_job_ids,
            extract_payload=p.extract_payload,
            detect_failed_reason=_detected,
            summarize_exc_info=_sum,
            make_logger=p.make_logger,
            redis_pubsub=p.redis_pubsub,
            redis_config_get=p.redis_config_get,
            redis_db_index=p.redis_db_index,
        )

    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_with_hint(),
    )
    w._handle_message(
        {
            "type": "pmessage",
            "pattern": "__keyspace@0__:rq:failed:digits",
            "channel": "__keyspace@0__:rq:failed:digits",
            "data": "zadd",
        }
    )
    # Message was summarized from hint
    assert any("SUM:HINT" in ev for _ch, ev in pub.items)


def test_scheduled_event_noop() -> None:
    pub = _Pub()
    st = _Store()
    w = NotificationWatcher(
        redis_url="redis://localhost/0",
        queues=("digits",),
        events_channel="digits:events",
        publisher=pub,
        store=st,
        ports=_ports_stub(failed=[], started=[], canceled=[]),
    )
    # Scheduled registry event should be ignored for publishing
    w._handle_message(
        {
            "type": "pmessage",
            "pattern": "__keyspace@0__:rq:scheduled:digits",
            "channel": "__keyspace@0__:rq:scheduled:digits",
            "data": "zadd",
        }
    )
    assert pub.items == []
