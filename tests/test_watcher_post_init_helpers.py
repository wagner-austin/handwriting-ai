from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def _rfu(url: str, *, decode_responses: bool = False) -> RedisClient:
    class _C:
        def publish(self, channel: str, message: str) -> int:
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return False

        def sadd(self, key: str, *members: str) -> int:
            return len(members)

    return _C()


def test_post_init_uses_provided_ports_provider() -> None:
    # Build a ports provider and ensure fw uses it verbatim
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
        redis_from_url=_rfu,
        rq_connect=_conn,
        rq_queue=_queue,
        rq_failed_registry=lambda _q: _dummy_reg(),
        rq_started_registry=lambda _q: _dummy_reg(),
        rq_stopped_registry=lambda _q: _dummy_reg(),
        rq_canceled_registry=lambda _q: _dummy_reg(),
        rq_fetch_job=lambda _c, _j: _dummy_job(),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )

    def provider() -> WatcherPorts:
        return ports

    fw = FWatcher("redis://", "digits", "digits:events", ports_provider=provider)
    assert fw.ports_provider is provider
    assert fw.publisher is not None and fw.store is not None


@dataclass
class _DummyJob:
    args: object = None
    exc_info: object = None
    started_at: object = None
    enqueued_at: object = None


def _dummy_job() -> RQJobProto:
    return _DummyJob()


class _DummyReg:
    def get_job_ids(self) -> list[str]:  # pragma: no cover - trivial
        return []


def _dummy_reg() -> RQRegistryProto:
    return _DummyReg()


def test_post_init_keeps_existing_pub_and_store() -> None:
    @dataclass
    class _Pub:
        def publish(self, channel: str, message: str) -> int:
            return 1

    @dataclass
    class _Store:
        def seen(self, job_id: str) -> bool:
            return False

        def mark(self, job_id: str) -> None:
            pass

    ports = WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=lambda u: _dummy_conn(),
        rq_queue=lambda c, n: object(),
        rq_failed_registry=lambda _q: _dummy_reg(),
        rq_started_registry=lambda _q: _dummy_reg(),
        rq_stopped_registry=lambda _q: _dummy_reg(),
        rq_canceled_registry=lambda _q: _dummy_reg(),
        rq_fetch_job=lambda _c, _j: _dummy_job(),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )
    pub = _Pub()
    store = _Store()
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)
    assert fw.publisher is pub
    assert fw.store is store


def _dummy_conn() -> RedisDebugClientProto:
    class _C:
        # pragma: no cover - trivial
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
            return []

        def hget(self, key: str, field: str) -> str | bytes | None:
            return None

    return _C()
