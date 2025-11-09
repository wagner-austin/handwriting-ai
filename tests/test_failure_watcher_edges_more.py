from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar

import pytest

import handwriting_ai.jobs.failure_watcher as mod
from handwriting_ai.jobs.failure_watcher import FailureWatcher as FWatcher


def test_summarize_exc_info_detects_sigkill_and_waitpid() -> None:
    msg1 = "Worker killed; received SIGKILL (signal 9) during processing\nMore..."
    out1 = mod._summarize_exc_info(msg1)
    assert "OOM kill detected" in out1 or "SIGKILL" in out1

    msg2 = "Work-horse terminated unexpectedly; waitpid returned 9; details.."
    out2 = mod._summarize_exc_info(msg2)
    assert "Worker killed by OS" in out2


def test_coerce_job_ids_handles_bad_bytes_and_success() -> None:
    bad = b"\xff\xfe\xff"  # invalid UTF-8 should be skipped
    good = b"ok"
    out = mod._coerce_job_ids([bad, good, "already_str"])
    assert out == ["ok", "already_str"]


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


def test_scan_once_continue_on_non_str_jid_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force list containing a non-string to cover the continue branch inside the loop
    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    class _Reg:
        def get_job_ids(self) -> list[str]:
            return ["unused"]

    def _fetch(_c: object, jid: str) -> object:
        class _J:
            args: ClassVar[list[dict[str, object]]] = [
                {"request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            exc_info: ClassVar[str] = "error"

        assert jid == "j_ok"
        return _J()

    # Stub rq functions
    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)

    def _mk_reg(_q: object) -> object:
        return _Reg()

    monkeypatch.setattr(mod, "_rq_failed_registry", _mk_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    # Force _coerce_job_ids to return one non-string and one valid string id
    def _force_ids(_items: object) -> list[object]:
        return [123, "j_ok"]

    monkeypatch.setattr(mod, "_coerce_job_ids", _force_ids, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        publisher=pub,
        store=store,
    )
    fw.scan_once()
    # Only the valid string id should be processed and published
    assert len(pub.items) == 1
    ch, msg = pub.items[0]
    assert ch == "digits:events"
    payload: dict[str, object] = json.loads(msg)
    assert payload.get("type") == "digits.train.failed.v1"
