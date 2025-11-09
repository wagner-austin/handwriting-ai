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

    def _failed_reg(_q: object) -> object:
        return _Reg(["j1"])  # one failed job

    def _started_reg(_q: object) -> object:
        return _Reg([])  # no started jobs

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
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
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
    # Second scan should no-op due to deduplication
    fw.scan_once()

    assert len(pub.items) == 1
    ch, msg = pub.items[0]
    assert ch == "digits:events"
    evt: dict[str, object] = json.loads(msg)
    assert evt.get("type") == "digits.train.failed.v1"
    assert evt.get("request_id") == "r"
    uid = evt.get("user_id")
    assert isinstance(uid, int) and uid == 5
    assert evt.get("model_id") == "m"
    msg_text = evt.get("message")
    assert isinstance(msg_text, str) and ("RuntimeError" in msg_text)
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

    def _started_reg(_q: object) -> object:
        return _Reg([])  # no started jobs

    def _fetch_fail(_c: object, jid: str) -> object:
        raise RuntimeError("missing job")

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch_fail, raising=True)

    fw = FWatcher(
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

    def _started_reg2(_q: object) -> object:
        return _Reg([])  # no started jobs

    def _fetch2(_c: object, _jid: str) -> object:
        return _Job(payload=[1, 2, 3], exc_info=None)

    monkeypatch.setattr(mod, "_rq_failed_registry", _reg2, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg2, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch2, raising=True)
    fw.scan_once()
    assert len(pub.items) == 1
    _, msg = pub.items[-1]
    evt2: dict[str, object] = json.loads(msg)
    assert evt2.get("type") == "digits.train.failed.v1"
    assert evt2.get("request_id") == ""
    uid2 = evt2.get("user_id")
    assert isinstance(uid2, int) and uid2 == 0
    assert evt2.get("model_id") == ""
    assert evt2.get("message") == "job failed"


def test_redis_publisher_and_store_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Client:
        def __init__(self) -> None:
            self._set: set[str] = set()

        def publish(self, channel: str, message: str) -> int:
            assert isinstance(channel, str) and isinstance(message, str)
            return 1

        def sismember(self, key: str, member: str) -> bool:
            return member in self._set

        def sadd(self, key: str, *members: str) -> int:
            for m in members:
                self._set.add(m)
            return len(members)

    _shared = _Client()

    def _fake_redis_from_url(*a: object, **k: object) -> object:
        return _shared

    monkeypatch.setattr(mod, "_redis_from_url", _fake_redis_from_url, raising=True)

    # Exercise default publisher/store via __post_init__ and methods
    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
    )
    assert fw.publisher is not None and fw.store is not None
    assert fw.publisher.publish("digits:events", "msg") == 1
    st = fw.store
    assert st is not None
    assert st.seen("x") is False
    st.mark("x")
    assert st.seen("x") is True

    # Failure branches for publisher/store
    def _raise(*_a: object, **_k: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_redis_from_url", _raise, raising=True)
    pub = mod._RedisPublisher("redis://")
    assert pub.publish("ch", "m") == 0
    store = mod._RedisProcessedStore("redis://", key="k")
    assert store.seen("id") is False
    # mark() should not raise when client fails
    store.mark("id")


def test_scan_once_skips_when_store_none(monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _Pub()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    class _Reg:
        def get_job_ids(self) -> list[str]:
            return ["a", "b"]

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)

    def _fake_reg(_q: object) -> object:
        return _Reg()

    def _started_reg(_q: object) -> object:
        return _Reg()

    monkeypatch.setattr(mod, "_rq_failed_registry", _fake_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        publisher=pub,
        store=_Store(),
    )
    # Force store None branch
    fw.store = None
    fw.scan_once()  # should not raise and should not publish
    assert pub.items == []


def test_scan_once_skips_non_string_job_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _Pub()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    class _Reg:
        def get_job_ids(self) -> list[object]:
            return [123]  # non-string id triggers skip branch

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)

    def _mk_reg(_q: object) -> object:
        return _Reg()

    def _started_reg(_q: object) -> object:
        return _Reg()

    monkeypatch.setattr(mod, "_rq_failed_registry", _mk_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        publisher=pub,
        store=_Store(),
    )
    fw.scan_once()
    assert pub.items == []


def test_scan_once_with_no_failed_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    class _Reg:
        def get_job_ids(self) -> list[str]:
            return []

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)

    def _mk_reg2(_q: object) -> object:
        return _Reg()

    def _started_reg(_q: object) -> object:
        return _Reg()

    monkeypatch.setattr(mod, "_rq_failed_registry", _mk_reg2, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        publisher=pub,
        store=store,
    )
    # Nothing to process, should do nothing and not error
    fw.scan_once()
    assert pub.items == [] and store.seen_ids == set()


def test_run_from_env_invokes_run_forever(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {"n": 0}

    class _OneShot(FWatcher):
        def run_forever(self) -> None:  # pragma: no cover - trivial wrapper
            called["n"] += 1

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("RQ__QUEUE", "digits")
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setenv("RQ_WATCHER_POLL_SECONDS", "2")

    monkeypatch.setattr(mod, "FailureWatcher", _OneShot, raising=True)
    mod.run_from_env()
    assert called["n"] == 1


def test_run_from_env_missing_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(RuntimeError):
        mod.run_from_env()


def test_extract_payload_branches() -> None:
    # args is None -> {}
    class _J1:
        args = None

    assert mod._extract_payload(_J1()) == {}

    # first arg is dict -> populated
    class _J2:
        args: ClassVar[list[dict[str, object]]] = [
            {"request_id": "r", "user_id": 1, "model_id": "m", "type": "t"}
        ]

    p = mod._extract_payload(_J2())
    assert p.get("request_id") == "r" and p.get("user_id") == 1 and p.get("model_id") == "m"

    # first arg not dict -> {}
    class _J3:
        args: ClassVar[list[int]] = [123, 456]

    assert mod._extract_payload(_J3()) == {}


def test_make_logger_direct_call() -> None:
    # Ensure the helper is covered
    lg = mod._make_logger()
    assert hasattr(lg, "info")


def test_scan_once_with_no_publisher_marks_only(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    pub = _Pub()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _reg(_q: object) -> object:
        return _Reg(["j9"])  # single failed job id

    def _started_reg(_q: object) -> object:
        return _Reg([])  # no started jobs

    def _fetch(_c: object, _jid: str) -> object:
        return _Job(payload={"request_id": "r9", "user_id": 2, "model_id": "m"}, exc_info=None)

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
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
    # Remove publisher to exercise false branch
    fw.publisher = None
    fw.scan_once()
    assert pub.items == []
    assert "j9" in store.seen_ids


def test_stuck_job_detection_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stuck jobs in StartedJobRegistry are detected after timeout."""
    import datetime as dt

    pub = _Pub()
    store = _Store()

    # Stub RQ functions
    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])  # no failed jobs

    def _started_reg(_q: object) -> object:
        return _Reg(["stuck1"])  # one stuck job

    # Job that started 40 minutes ago (past 30 min timeout)
    started_time = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=40)

    class _StuckJob:
        def __init__(self) -> None:
            self.args = [
                {
                    "type": "digits.train.v1",
                    "request_id": "req_stuck",
                    "user_id": 99,
                    "model_id": "model_stuck",
                }
            ]
            self.started_at = started_time
            self.enqueued_at = None
            self.exc_info = None

    def _fetch(_c: object, jid: str) -> object:
        assert jid == "stuck1"
        return _StuckJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        stuck_job_timeout_s=1800.0,  # 30 minutes
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should publish stuck job event
    assert len(pub.items) == 1
    ch, msg = pub.items[0]
    assert ch == "digits:events"
    evt: dict[str, object] = json.loads(msg)
    assert evt.get("type") == "digits.train.failed.v1"
    assert evt.get("request_id") == "req_stuck"
    uid = evt.get("user_id")
    assert isinstance(uid, int) and uid == 99
    assert evt.get("model_id") == "model_stuck"
    msg_text = evt.get("message")
    assert isinstance(msg_text, str)
    assert "stuck in started state" in msg_text
    assert "40 minutes" in msg_text
    assert "OOM" in msg_text or "signal 9" in msg_text
    assert "stuck1" in store.seen_ids


def test_stuck_job_no_timeout_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that jobs under timeout threshold are not marked as stuck."""
    import datetime as dt

    pub = _Pub()
    store = _Store()

    # Stub RQ functions
    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])  # no failed jobs

    def _started_reg(_q: object) -> object:
        return _Reg(["running1"])  # one running job

    # Job that started 10 minutes ago (under 30 min timeout)
    started_time = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=10)

    class _RunningJob:
        def __init__(self) -> None:
            self.args = [
                {
                    "type": "digits.train.v1",
                    "request_id": "req_running",
                    "user_id": 88,
                    "model_id": "model_running",
                }
            ]
            self.started_at = started_time
            self.enqueued_at = None
            self.exc_info = None

    def _fetch(_c: object, jid: str) -> object:
        assert jid == "running1"
        return _RunningJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        poll_interval_s=0.01,
        stuck_job_timeout_s=1800.0,  # 30 minutes
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should NOT publish anything - job is still running normally
    assert len(pub.items) == 0
    # Job should NOT be marked as seen/processed
    assert "running1" not in store.seen_ids


def test_stuck_job_fetch_error_marks_seen(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch errors for stuck jobs are handled gracefully."""
    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])

    def _started_reg(_q: object) -> object:
        return _Reg(["stuck_missing"])

    def _fetch(_c: object, jid: str) -> object:
        raise RuntimeError("job missing from redis")

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should mark as seen even though fetch failed
    assert "stuck_missing" in store.seen_ids
    # Should not publish
    assert len(pub.items) == 0


def test_stuck_job_no_timestamp_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stuck jobs without timestamps are skipped."""

    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])

    def _started_reg(_q: object) -> object:
        return _Reg(["no_ts"])

    class _NoTimestampJob:
        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = None
            self.enqueued_at = None

    def _fetch(_c: object, jid: str) -> object:
        return _NoTimestampJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should not mark as seen (early return before marking)
    assert "no_ts" not in store.seen_ids
    assert len(pub.items) == 0


def test_stuck_job_naive_timezone_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stuck jobs with naive timestamps are handled."""
    import datetime as dt

    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])

    def _started_reg(_q: object) -> object:
        return _Reg(["naive_tz"])

    # Naive datetime (no timezone)
    naive_time = dt.datetime.now() - dt.timedelta(minutes=40)

    class _NaiveJob:
        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = naive_time
            self.enqueued_at = None

    def _fetch(_c: object, jid: str) -> object:
        return _NaiveJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        stuck_job_timeout_s=1800.0,
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should handle naive timezone and publish stuck event
    assert len(pub.items) == 1
    assert "naive_tz" in store.seen_ids


def test_stuck_job_fallback_to_enqueued_at(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that enqueued_at is used when started_at is None."""
    import datetime as dt

    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])

    def _started_reg(_q: object) -> object:
        return _Reg(["enqueued"])

    enqueued_time = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=40)

    class _EnqueuedJob:
        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = None
            self.enqueued_at = enqueued_time

    def _fetch(_c: object, jid: str) -> object:
        return _EnqueuedJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        stuck_job_timeout_s=1800.0,
        publisher=pub,
        store=store,
    )
    fw.scan_once()

    # Should use enqueued_at and detect as stuck
    assert len(pub.items) == 1
    assert "enqueued" in store.seen_ids


def test_stuck_job_no_publisher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stuck jobs without publisher still mark as seen."""
    import datetime as dt

    pub = _Pub()
    store = _Store()

    def _conn(_url: str) -> object:
        return object()

    def _queue(_c: object, _n: str) -> object:
        return object()

    def _failed_reg(_q: object) -> object:
        return _Reg([])

    def _started_reg(_q: object) -> object:
        return _Reg(["stuck_no_pub"])

    started_time = dt.datetime.now(tz=dt.UTC) - dt.timedelta(minutes=40)

    class _StuckJob:
        def __init__(self) -> None:
            self.args = [
                {"type": "digits.train.v1", "request_id": "r", "user_id": 1, "model_id": "m"}
            ]
            self.started_at = started_time
            self.enqueued_at = None

    def _fetch(_c: object, jid: str) -> object:
        return _StuckJob()

    monkeypatch.setattr(mod, "_rq_connect", _conn, raising=True)
    monkeypatch.setattr(mod, "_rq_queue", _queue, raising=True)
    monkeypatch.setattr(mod, "_rq_failed_registry", _failed_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_started_registry", _started_reg, raising=True)
    monkeypatch.setattr(mod, "_rq_fetch_job", _fetch, raising=True)

    fw = FWatcher(
        redis_url="redis://localhost:6379/0",
        queue_name="digits",
        events_channel="digits:events",
        stuck_job_timeout_s=1800.0,
        publisher=pub,  # Provide publisher initially
        store=store,
    )
    # Remove publisher to test the no-publisher warning path
    fw.publisher = None
    fw.scan_once()

    # Should mark as seen even without publisher
    assert "stuck_no_pub" in store.seen_ids
    assert len(pub.items) == 0
