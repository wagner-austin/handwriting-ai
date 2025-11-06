from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from handwriting_ai.jobs.failure_watcher import FailureWatcher as _FW
import handwriting_ai.jobs.failure_watcher as mod


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
    def __init__(self, payload: object, exc_info: str | None) -> None:
        self.args = [payload]
        self.exc_info = exc_info


class _Reg:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def get_job_ids(self) -> list[str]:
        return list(self._ids)


def test_failure_watcher_publishes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _Pub()
    store = _Store()

    # Stub RQ functions
    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _reg(_q: object) -> object:
        return _Reg(["j1"])  # one failed job

    def _fetch(_c: object, jid: str) -> object:
        assert jid == "j1"
        payload = {
            "type": "digits.train.v1",
            "request_id": "r",
            "user_id": 5,
            "model_id": "m",
        }
        return _Job(payload, exc_info="RuntimeError: boom\ntrace…")

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = _FW(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        publisher=pub,
        store=store,
    )
    fw.scan_once()
    # Second scan should no-op due to deduplication
    fw.scan_once()

    assert len(pub.items) == 1
    ch, msg = pub.items[0]
    assert ch == "digits:events"
    evt = json.loads(msg)
    assert evt["type"] == "digits.train.failed.v1"
    assert evt["request_id"] == "r" and int(evt["user_id"]) == 5 and evt["model_id"] == "m"
    assert "RuntimeError" in evt["message"]
    assert "j1" in store.seen_ids


def test_failure_watcher_missing_payload_and_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _Pub()
    store = _Store()

    # First run: fetch raises -> store.mark called, no publish
    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _reg(_q: object) -> object:
        return _Reg(["j2"])  # one failed job

    def _fetch_fail(_c: object, jid: str) -> object:
        raise RuntimeError("missing job")

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch_fail, raising=True)

    fw = _FW(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        publisher=pub,
        store=store,
    )
    fw.scan_once()
    assert pub.items == []
    assert "j2" in store.seen_ids

    # Second run: job exists but payload shape invalid -> still published with defaults
    def _reg2(_q: object) -> object:
        return _Reg(["j3"])  # another failed job

    def _fetch2(_c: object, _jid: str) -> object:
        return _Job(payload=[1, 2, 3], exc_info=None)

    monkeypatch.setattr(mod, "_rq_failed_registry", _reg2, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch2, raising=True)
    fw.scan_once()
    assert len(pub.items) == 1
    _, msg = pub.items[-1]
    evt = json.loads(msg)
    assert evt["type"] == "digits.train.failed.v1"
    assert evt["request_id"] == "" and int(evt["user_id"]) == 0 and evt["model_id"] == ""
    assert evt["message"] == "job failed"

