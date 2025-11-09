from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

import handwriting_ai.jobs.failure_watcher as mod
from handwriting_ai.jobs.failure_watcher import FailureWatcher as FWatcher


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
    args: ClassVar[list[dict[str, object]]] = [
        {"request_id": "r-oom", "user_id": 7, "model_id": "m"}
    ]
    exc_info: ClassVar[None] = None


def test_failure_watcher_uses_redis_reason_to_detect_oom(monkeypatch: pytest.MonkeyPatch) -> None:
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

    def _mk_conn(_url: str) -> object:
        return conn

    class _FailedReg:
        def get_job_ids(self) -> list[str]:
            return [jid]

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _reg(_q: object) -> object:
        return _FailedReg()

    def _started_reg(_q: object) -> object:
        return _Reg([])  # no started jobs

    def _fetch(_c: object, _jid: str) -> object:
        return _JobNoExc()

    monkeypatch.setattr(mod, "_rq_connect", _mk_conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        publisher=pub,
        store=store,
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
