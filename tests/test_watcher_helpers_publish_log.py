from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.jobs.watcher.ports import WatcherPorts
from handwriting_ai.jobs.watcher.watcher import FailureWatcher as FWatcher

if TYPE_CHECKING:
    from handwriting_ai.jobs.watcher.adapters import (
        RedisClient,
        RedisDebugClientProto,
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


def _rfu(url: str, *, decode_responses: bool = False) -> RedisClient:
    class _C:
        def publish(self, channel: str, message: str) -> int:
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return False

        def sadd(self, key: str, *members: str) -> int:
            return len(members)

    return _C()


def _dummy_conn() -> RedisDebugClientProto:
    class _C:
        # pragma: no cover - trivial
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]:
            return []

        def hget(self, key: str, field: str) -> str | bytes | None:
            return None

    return _C()


def _ports() -> WatcherPorts:
    return WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=lambda u: _dummy_conn(),
        rq_queue=lambda c, n: object(),
        rq_failed_registry=lambda _q: _Reg([]),
        rq_started_registry=lambda _q: _Reg([]),
        rq_stopped_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=lambda _c, _j: _Job({}, None),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
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


class _Reg:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def get_job_ids(self) -> list[str]:  # pragma: no cover - trivial
        return list(self._ids)


def test_log_registry_processing_branches() -> None:
    fw = FWatcher("redis://", "digits", "digits:events", ports=_ports())
    logger = logging.getLogger("handwriting_ai.testhelpers")
    captured: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    h = _H()
    logger.addHandler(h)
    try:
        fw._log_registry_processing(logger, "failed", [])
        assert captured == []
        fw._log_registry_processing(logger, "failed", ["a", "b"])
        assert len(captured) == 1
        assert "rq_failure_watcher_processing_failed" in captured[0].getMessage()
    finally:
        logger.removeHandler(h)


def test_publish_and_mark_with_pub_and_without_pub() -> None:
    import handwriting_ai.events.digits as ev

    # With publisher present
    pub = _Pub()
    store = _Store()
    fw = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=pub,
        store=store,
        ports=_ports(),
    )
    evt = ev.failed(
        ev.Context(request_id="r", user_id=1, model_id="m", run_id=None),
        error_kind="system",
        message="x",
    )
    fw._publish_and_mark(evt, "j1")
    assert "j1" in store.seen_ids
    assert len(pub.items) == 1
    decoded: dict[str, object] = json.loads(pub.items[0][1])
    assert decoded.get("type") == "digits.train.failed.v1"

    # Without publisher
    pub2 = _Pub()
    store2 = _Store()
    fw2 = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=None,
        store=store2,
        ports=_ports(),
    )
    fw2._publish_and_mark(evt, "j2")
    assert "j2" in store2.seen_ids and pub2.items == []


def test_publish_and_mark_store_none_early_return() -> None:
    import handwriting_ai.events.digits as ev

    pub = _Pub()
    fw = FWatcher(
        "redis://",
        "digits",
        "digits:events",
        publisher=pub,
        store=_Store(),
        ports=_ports(),
    )
    # Remove store to exercise early return in helper
    fw.store = None
    evt = ev.failed(
        ev.Context(request_id="r2", user_id=2, model_id="m2", run_id=None),
        error_kind="system",
        message="x",
    )
    # Should not raise
    fw._publish_and_mark(evt, "j3")
