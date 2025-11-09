from __future__ import annotations

import json
from dataclasses import dataclass, field
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


class _Reg:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def get_job_ids(self) -> list[str]:
        return list(self._ids)


class _JobNoExc:
    args: object
    exc_info: object
    started_at: object
    enqueued_at: object

    def __init__(self) -> None:
        self.args = [{"request_id": "r-oom", "user_id": 7, "model_id": "m"}]
        self.exc_info = None
        self.started_at = None
        self.enqueued_at = None


def _build_ports_for_oom(conn: RedisDebugClientProto, jid: str) -> WatcherPorts:
    def _mk_conn(_url: str) -> RedisDebugClientProto:
        return conn

    def _queue(_c: RedisDebugClientProto, _n: str) -> RQQueueProto:
        return object()

    class _FailedReg:
        def get_job_ids(self) -> list[str]:
            return [jid]

    def _reg(_q: RQQueueProto) -> RQRegistryProto:
        return _FailedReg()

    def _started_reg(_q: RQQueueProto) -> RQRegistryProto:
        return _Reg([])

    def _fetch(_c: RedisDebugClientProto, _jid: str) -> RQJobProto:
        return _JobNoExc()

    def _rfu(url: str, *, decode_responses: bool = False) -> RedisClient:
        class _Dummy:
            def publish(self, channel: str, message: str) -> int:
                return 1

            def sismember(self, key: str, member: str) -> bool:
                return False

            def sadd(self, key: str, *members: str) -> int:
                return 0

        return _Dummy()

    return WatcherPorts(
        redis_from_url=_rfu,
        rq_connect=_mk_conn,
        rq_queue=_queue,
        rq_failed_registry=_reg,
        rq_started_registry=_started_reg,
        rq_stopped_registry=lambda _q: _Reg([]),
        rq_canceled_registry=lambda _q: _Reg([]),
        rq_fetch_job=_fetch,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
    )


def test_failure_watcher_uses_redis_reason_to_detect_oom() -> None:
    pub = _Pub()
    store = _Store()

    class _Conn:
        def __init__(self) -> None:
            self._h: dict[tuple[str, str], bytes] = {}

        # Simulate Redis.zrange used by diagnostic logging
        def zrange(self, key: str, start: int, end: int) -> list[str]:  # pragma: no cover - trivial
            return []

        # Provide hget for job hash fields
        def hget(self, key: str, field: str) -> bytes | None:
            return self._h.get((key, field))

    conn = _Conn()
    # Fill the job hash with a failure reason typical of SIGKILL/oom kills
    jid = "job-9"
    conn._h[(f"rq:job:{jid}", "failed_reason")] = (
        b"Work-horse terminated unexpectedly; waitpid returned 9 (signal 9); "
    )

    ports = _build_ports_for_oom(conn, jid)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        publisher=pub,
        store=store,
        ports=ports,
    )
    fw.scan_once()

    # One event published with OOM/SIGKILL classification
    assert len(pub.items) == 1
    ch, msg = pub.items[0]
    assert ch == "digits:events"
    evt: dict[str, object] = json.loads(msg)
    assert evt.get("type") == "digits.train.failed.v1"
    m: object = evt.get("message")
    assert isinstance(m, str) and ("SIGKILL" in m or "signal 9" in m or "OOM" in m)
