from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.jobs.watcher.ports import WatcherPorts
from handwriting_ai.jobs.watcher.publisher import RedisPublisher
from handwriting_ai.jobs.watcher.store import RedisProcessedStore
from handwriting_ai.jobs.watcher.watcher import FailureWatcher as FWatcher

if TYPE_CHECKING:
    from handwriting_ai.jobs.watcher.adapters import (
        RedisClient,
        RedisDebugClientProto,
        RQJobProto,
        RQQueueProto,
        RQRegistryProto,
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


class _Reg:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def get_job_ids(self) -> list[str]:
        return list(self._ids)


def _dummy_conn() -> RedisDebugClientProto:
    class _C:
        # pragma: no cover - trivial
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
            return []

        def hget(self, key: str, field: str) -> str | bytes | None:
            return None

    return _C()


def _rfu_ok(url: str, *, decode_responses: bool = False) -> RedisClient:
    class _Client:
        def publish(self, channel: str, message: str) -> int:
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return False

        def sadd(self, key: str, *members: str) -> int:
            return len(members)

    return _Client()


def test_publisher_error_path_direct() -> None:
    class _RFU:
        def __call__(self, url: str, *, decode_responses: bool = False) -> RedisClient:
            class _C:
                def publish(self, channel: str, message: str) -> int:
                    raise RuntimeError("boom")

                def sismember(self, key: str, member: str) -> bool:  # pragma: no cover - unused
                    return False

                def sadd(self, key: str, *members: str) -> int:  # pragma: no cover - unused
                    return 0

            return _C()

    pub = RedisPublisher("redis://", redis_factory=_RFU())
    with pytest.raises(RuntimeError):
        pub.publish("ch", "msg")


def test_watcher_publish_raises_and_marks() -> None:
    # Failed registry with one id; publisher raises; store still marks in finally
    class _RFUFail:
        def __call__(self, url: str, *, decode_responses: bool = False) -> RedisClient:
            class _C:
                def publish(self, channel: str, message: str) -> int:
                    raise RuntimeError("boom")

                def sismember(self, key: str, member: str) -> bool:
                    return False

                def sadd(self, key: str, *members: str) -> int:
                    return len(members)

            return _C()

    store = _Store()

    class _Conn:
        # pragma: no cover - trivial stubs
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
            return []

        def hget(self, key: str, field: str) -> str | bytes | None:
            return None

    def _conn(_url: str) -> RedisDebugClientProto:
        return _Conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["jid1"])

    def _fetch(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _Job({"request_id": "r1", "user_id": 1, "model_id": "m"}, exc_info=None)

    ports = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )

    pub = RedisPublisher("redis://", redis_factory=_RFUFail())
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    with pytest.raises(RuntimeError):
        fw.scan_once()
    assert "jid1" in store.seen_ids


def test_post_init_ports_fixed_provider() -> None:
    # Providing ports should result in a fixed provider being used
    def _conn(_url: str) -> RedisDebugClientProto:
        class _C:
            def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
                return []

            def hget(self, key: str, field: str) -> str | bytes | None:
                return None

        return _C()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    ports = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=lambda _q: _Reg([]),
        rq_started_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=lambda _c, _j: _Job({}, None),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw = FWatcher("redis://", "digits", "digits:events", ports=ports)
    assert fw.ports_provider is not None
    assert fw.ports_provider() is ports
    assert fw.publisher is not None and fw.store is not None


def test_stuck_job_paths() -> None:
    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> RedisDebugClientProto:
        class _C:
            def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
                return []

            def hget(self, key: str, field: str) -> str | bytes | None:
                return None

        return _C()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg([])

    # a) stuck beyond timeout publishes
    started_time = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=40)

    class _Stuck:
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "req", "user_id": 9, "model_id": "m"}
            ]
            self.started_at = started_time
            self.enqueued_at = None
            self.exc_info = None

    def _started_reg1(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["s1"])

    def _fetch1(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _Stuck()

    ports1 = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg1,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch1,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw1 = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports1)
    fw1.scan_once()
    assert len(pub.items) >= 1 and "s1" in store.seen_ids

    # b) under-timeout returns and not marked
    pub.items.clear()
    store.seen_ids.clear()

    class _Running:
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=10)
            self.enqueued_at = None
            self.exc_info = None

    def _started_reg2(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["s2"])

    def _fetch2(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _Running()

    ports2 = ports1.__class__(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg2,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch2,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw2 = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports2)
    fw2.scan_once()
    assert len(pub.items) == 0 and "s2" not in store.seen_ids

    # c) no timestamps -> early return and not marked
    pub.items.clear()
    store.seen_ids.clear()

    class _NoTs:
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = None
            self.enqueued_at = None
            self.exc_info = None

    def _started_reg3(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["s3"])

    def _fetch3(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _NoTs()

    ports3 = ports1.__class__(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg3,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch3,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw3 = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports3)
    fw3.scan_once()
    assert len(pub.items) == 0 and "s3" not in store.seen_ids

    # d) naive datetime handled and publishes
    pub.items.clear()
    store.seen_ids.clear()
    naive_time = dt.datetime.now() - dt.timedelta(minutes=40)

    class _Naive:
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = naive_time
            self.enqueued_at = None
            self.exc_info = None

    def _started_reg4(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["s4"])

    def _fetch4(_c: RedisDebugClientProto, _j: str) -> RQJobProto:
        return _Naive()

    ports4 = ports1.__class__(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg4,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch4,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw4 = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports4)
    fw4.scan_once()
    assert len(pub.items) >= 1 and "s4" in store.seen_ids


def test_stuck_job_fetch_error_raises() -> None:
    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> RedisDebugClientProto:
        return _dummy_conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg([])

    def _started_reg(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["sid"])

    def _fetch_fail(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        raise RuntimeError("missing job")

    ports = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch_fail,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    with pytest.raises(RuntimeError):
        fw.scan_once()


def test_stopped_and_canceled_seen_early_return() -> None:
    pub = _Pub()
    store = _Store()
    store.seen_ids.add("s_seen")
    store.seen_ids.add("c_seen")

    def _conn(_url: str) -> RedisDebugClientProto:
        return _dummy_conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg([])

    def _canceled_reg(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["c_seen"])

    ports = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=lambda _q: _Reg([]),
        rq_canceled_registry=_canceled_reg,
        rq_fetch_job=lambda _c, _j: _Job({}, None),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    fw.scan_once()
    # No publish because already seen
    assert pub.items == []


def test_stuck_seen_early_return_and_enqueued_fallback() -> None:
    pub = _Pub()
    store = _Store()
    store.seen_ids.add("sx")

    def _conn(_url: str) -> RedisDebugClientProto:
        return _dummy_conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg([])

    # First: early return when already seen
    ports_seen = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=lambda _q: _Reg(["sx"]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=lambda _c, _j: _Job({}, None),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw_seen = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=pub,
        store=store,
        ports=ports_seen,
    )
    fw_seen.scan_once()
    assert pub.items == []

    # Second: started_at None, enqueued_at set beyond timeout -> fallback path
    import datetime as dt

    class _Enq:
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

        def __init__(self) -> None:
            self.args = [{"request_id": "re", "user_id": 1, "model_id": "m"}]
            self.exc_info = None
            self.started_at = None
            self.enqueued_at = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=40)

    def _started_reg_enq(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["se"])  # new id

    ports_enq = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started_reg_enq,
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=lambda _c, _j: _Enq(),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw_enq = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=pub,
        store=store,
        ports=ports_enq,
    )
    fw_enq.scan_once()
    # Should publish and mark.
    import json as _json

    types: list[str] = []
    for _, payload in pub.items:
        dec: dict[str, object] = _json.loads(payload)
        t = dec.get("type")
        if isinstance(t, str):
            types.append(t)
    assert "digits.train.failed.v1" in types


def test_store_mark_error_path_direct() -> None:
    class _RFU:
        def __call__(self, url: str, *, decode_responses: bool = False) -> RedisClient:
            class _C:
                def publish(self, channel: str, message: str) -> int:  # pragma: no cover - unused
                    return 1

                def sismember(self, key: str, member: str) -> bool:  # pragma: no cover - unused
                    return False

                def sadd(self, key: str, *members: str) -> int:
                    raise RuntimeError("boom")

            return _C()

    store = RedisProcessedStore("redis://", key="k", redis_factory=_RFU())
    with pytest.raises(RuntimeError):
        store.mark("id")


def test_failed_reason_raiser_propagates() -> None:
    # detect_failed_reason raising should propagate in failed-job processing
    pub = _Pub()
    store = _Store()

    class _Conn:
        # pragma: no cover - trivial stubs
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
            return []

        def hget(self, key: str, field: str) -> str | bytes | None:
            return None

    def _conn(_url: str) -> RedisDebugClientProto:
        return _Conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["fj1"])

    def _fetch(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _Job({"request_id": "r", "user_id": 1, "model_id": "m"}, exc_info=None)

    def _raise_detect(_c: object, _j: str) -> str | None:
        raise RuntimeError("detect failed")

    ports = WatcherPorts(
        redis_from_url=_rfu_ok,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=_raise_detect,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    with pytest.raises(RuntimeError):
        fw.scan_once()
