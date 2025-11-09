from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.jobs.watcher.ports import WatcherPorts
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


class _Conn:
    def zrange(self, _key: str, _start: int, _end: int) -> list[str]:  # pragma: no cover - trivial
        return []

    def hget(self, _key: str, _field: str) -> str | None:  # pragma: no cover - default
        return None


def _make_ports(
    *,
    failed_reg: RQRegistryProto | None = None,
    started_reg: RQRegistryProto | None = None,
    canceled_reg: RQRegistryProto | None = None,
    fetch_job: Callable[[RedisDebugClientProto, str], RQJobProto] | None = None,
) -> WatcherPorts:
    def _rfu(url: str, *, decode_responses: bool = False) -> RedisClient:
        class _Dummy:
            def publish(self, channel: str, message: str) -> int:
                return 1

            def sismember(self, key: str, member: str) -> bool:
                return False

            def sadd(self, key: str, *members: str) -> int:
                return 0

        return _Dummy()

    conn_default: RedisDebugClientProto = _Conn()
    fr: RQRegistryProto = failed_reg if failed_reg is not None else _Reg([])
    sr: RQRegistryProto = started_reg if started_reg is not None else _Reg([])
    cancr: RQRegistryProto = canceled_reg if canceled_reg is not None else _Reg([])

    def _conn(_url: str) -> RedisDebugClientProto:
        return conn_default

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()  # RQQueueProto has no required members

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return fr

    def _started(_q: RQQueueProto) -> RQRegistryProto:
        return sr

    def _stopped(_q: RQQueueProto) -> RQRegistryProto:
        return stopr

    def _canceled(_q: RQQueueProto) -> RQRegistryProto:
        return cancr

    def _fetch(_c: RedisDebugClientProto, jid: str) -> RQJobProto:
        assert fetch_job is not None
        return fetch_job(_c, jid)

    return WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=_failed,
        rq_started_registry=_started,
        rq_canceled_registry=_canceled,
        rq_fetch_job=_fetch,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )


def test_canceled_job_publishes() -> None:
    pub = _Pub()
    store = _Store()

    def _fetch(_c: RedisDebugClientProto, jid: str) -> RQJobProto:
        assert jid == "c1"
        payload = {"type": "digits.train.v1", "request_id": "r3", "user_id": 9, "model_id": "m3"}
        return _Job(payload, exc_info=None)

    ports = _make_ports(
        failed_reg=_Reg([]),
        started_reg=_Reg([]),
        canceled_reg=_Reg(["c1"]),
        fetch_job=lambda c, j: _fetch(c, j),
    )

    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    fw.scan_once()
    assert len(pub.items) == 1
    evt: dict[str, object] = json.loads(pub.items[0][1])
    assert evt.get("type") == "digits.train.failed.v1"
    assert evt.get("error_kind") == "user"
    assert evt.get("message") == "Job canceled"




def test_stopped_and_canceled_fetch_errors_raise() -> None:
    pub = _Pub()
    store = _Store()

    def _fetch_fail(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        raise RuntimeError("missing job")

    # Canceled
    ports_c = _make_ports(
        failed_reg=_Reg([]),
        started_reg=_Reg([]),
        canceled_reg=_Reg(["c2"]),
        fetch_job=_fetch_fail,
    )
    fw_c = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=pub,
        store=store,
        ports=ports_c,
    )
    with pytest.raises(RuntimeError):
        fw_c.scan_once()


def test_stopped_and_canceled_store_none_short_circuit() -> None:
    pub = _Pub()

    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=None)
    # Ensure store is None (constructor populates default when None is passed)
    fw.store = None
    # Directly call internal method to exercise early return
    c = _Conn()
    fw._process_canceled_job(c, "jid")
    assert pub.items == []


def test_stopped_and_canceled_no_publisher_paths() -> None:
    store = _Store()

    def _fetch_ok(_c: RedisDebugClientProto, jid: str) -> RQJobProto:
        payload = {"type": "digits.train.v1", "request_id": jid, "user_id": 1, "model_id": "m"}
        return _Job(payload, exc_info=None)

    ports = _make_ports(fetch_job=_fetch_ok)

    fw = FWatcher("redis://", "digits", "digits:events", publisher=None, store=store, ports=ports)
    # Ensure publisher remains None (constructor populates default otherwise)
    fw.publisher = None
    # Call internal handler directly with no publisher
    c = _Conn()
    fw._process_canceled_job(c, "jidY")
    # Should be marked; nothing published
    assert "jidY" in store.seen_ids


def test_store_and_publisher_factory_fail_raise() -> None:
    # Cover error paths in store.seen/mark and publisher.publish via watcher
    class _RFU:
        def __call__(self, url: str, *, decode_responses: bool = False) -> RedisClient:
            raise RuntimeError("boom")

    def _conn(_url: str) -> RedisDebugClientProto:
        return _Conn()

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    def _failed(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg(["x1"])  # one failed job

    def _fetch(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _Job({"request_id": "r", "user_id": 1, "model_id": "m"}, exc_info=None)

    ports = WatcherPorts(
        redis_from_url=_RFU(),
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

    # Default store/publisher created via __post_init__ will use failing RFU; scan_once should raise
    fw = FWatcher("redis://", "digits", "digits:events", publisher=None, store=None, ports=ports)
    with pytest.raises(RuntimeError):
        fw.scan_once()
