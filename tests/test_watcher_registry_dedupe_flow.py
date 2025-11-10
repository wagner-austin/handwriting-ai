from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import handwriting_ai.jobs.watcher.logic as logic
from handwriting_ai.jobs.watcher.ports import WatcherPorts
from handwriting_ai.jobs.watcher.watcher import FailureWatcher as FWatcher


@dataclass
class _Store:
    seen_ids: set[str]

    def seen(self, job_id: str) -> bool:
        return job_id in self.seen_ids

    def mark(self, job_id: str) -> None:
        self.seen_ids.add(job_id)


class _Reg:
    def __init__(self, getter: Callable[[], list[str]]):
        self._getter = getter

    def get_job_ids(self) -> list[str]:
        return list(self._getter())


def _rfu(url: str, *, decode_responses: bool = False) -> _RedisClient:
    class _C:
        def publish(self, channel: str, message: str) -> int:
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return False

        def sadd(self, key: str, *members: str) -> int:
            return len(members)

    return _C()


def _dummy_conn() -> _RedisDebugClientProto:
    class _C:
        def zrange(self, key: str, start: int, end: int) -> list[str]:
            return []

        def hget(self, key: str, field: str) -> str | None:
            return None

    return _C()


def test_started_then_later_failed_not_blocked_by_seen() -> None:
    # First scan: started contains ID, failed empty; should NOT mark
    # Second scan: failed contains ID; should publish+mark once
    jid = "job-1"
    phase = {"n": 0}

    def _failed_ids() -> list[str]:
        return [jid] if phase["n"] >= 1 else []

    def _started_ids() -> list[str]:
        return [jid] if phase["n"] == 0 else []

    ports = WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=lambda u: _dummy_conn(),
        rq_queue=lambda c, n: object(),
        rq_failed_registry=lambda _q: _Reg(_failed_ids),
        rq_started_registry=lambda _q: _Reg(_started_ids),
        rq_canceled_registry=lambda _q: _Reg(lambda: []),
        rq_fetch_job=lambda _c, _j: _Job({}, None),
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )

    pub = _Pub()
    store = _Store(seen_ids=set())
    fw = FWatcher("redis://", "digits", "digits:events", publisher=pub, store=store, ports=ports)

    # Scan 1: started only — no mark
    fw.scan_once()
    assert jid not in store.seen_ids

    # Scan 2: failed — publish+mark
    phase["n"] = 1
    fw.scan_once()
    assert jid in store.seen_ids and len(pub.items) == 1


@dataclass
class _Pub:
    items: list[tuple[str, str]] = field(default_factory=list)

    def publish(self, channel: str, message: str) -> int:
        self.items.append((channel, message))
        return 1


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


if TYPE_CHECKING:
    from handwriting_ai.jobs.watcher.adapters import RedisClient as _RedisClient
    from handwriting_ai.jobs.watcher.adapters import (
        RedisDebugClientProto as _RedisDebugClientProto,
    )
