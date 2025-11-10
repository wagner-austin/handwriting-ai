from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.events import digits as ev
from handwriting_ai.training.mnist_train import TrainConfig


def _quick_training(cfg: TrainConfig) -> Path:
    root = Path(tempfile.mkdtemp())
    man: dict[str, object] = {
        "schema_version": "v1.1",
        "model_id": cfg.model_id,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "hash",
        "val_acc": 0.5,
        "temperature": 1.0,
    }
    (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return root


def test_process_train_job_keyboard_interrupt_publishes_interrupted_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pub:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def publish(self, channel: str, message: str) -> int:
            self.sent.append((channel, message))
            return 1

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p, raising=True)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_run_training", _quick_training, raising=True)

    def _pub_safe(ctx: dj._Context, event: ev.EventV1) -> None:  # use concrete types; no ignores
        if isinstance(event, dict) and event.get("type") == "digits.train.completed.v1":
            raise KeyboardInterrupt()
        if ctx.publisher is not None:
            ctx.publisher.publish(ctx.channel, ev.encode_event(event))

    monkeypatch.setattr(dj, "_publish_event_safe", _pub_safe, raising=True)

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r-kb",
        "user_id": 7,
        "model_id": "m-kb",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(KeyboardInterrupt):
        dj.process_train_job(payload)

    joined = "\n".join([m for _, m in p.sent])
    assert "digits.train.started.v1" in joined
    assert "digits.train.artifact.v1" in joined
    assert "digits.train.interrupted.v1" in joined


def test_progress_emitter_emit_batch_logs_and_raises_on_generic_exception() -> None:
    class _BadPub:
        def publish(self, channel: str, message: str) -> int:
            raise Exception("boom")

    em = dj._ProgressEmitter(
        publisher=_BadPub(),
        channel="ch",
        request_id="r",
        user_id=1,
        model_id="m",
        total_epochs=1,
    )
    metrics = ev.BatchMetrics(
        epoch=1,
        total_epochs=1,
        batch=1,
        total_batches=1,
        batch_loss=0.1,
        batch_acc=0.9,
        avg_loss=0.1,
        samples_per_sec=100.0,
        main_rss_mb=100,
        workers_rss_mb=50,
        worker_count=2,
        cgroup_usage_mb=500,
        cgroup_limit_mb=1000,
        cgroup_pct=50.0,
        anon_mb=200,
        file_mb=150,
    )
    with pytest.raises(Exception, match="boom"):
        em.emit_batch(metrics)


def test_progress_emitter_emit_best_raises_on_value_error() -> None:
    class _BadPub:
        def publish(self, channel: str, message: str) -> int:
            raise ValueError("bad")

    em = dj._ProgressEmitter(
        publisher=_BadPub(),
        channel="ch",
        request_id="r",
        user_id=1,
        model_id="m",
        total_epochs=1,
    )
    with pytest.raises(ValueError, match="bad"):
        em.emit_best(epoch=1, val_acc=0.5)


def test_progress_emitter_emit_epoch_raises_on_os_error() -> None:
    class _BadPub:
        def publish(self, channel: str, message: str) -> int:
            raise OSError("nope")

    em = dj._ProgressEmitter(
        publisher=_BadPub(),
        channel="ch",
        request_id="r",
        user_id=1,
        model_id="m",
        total_epochs=1,
    )
    with pytest.raises(OSError, match="nope"):
        em.emit_epoch(epoch=1, total_epochs=1, train_loss=0.1, val_acc=0.2, time_s=0.1)


def test_emit_failed_with_payload_mapping_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Pub:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def publish(self, channel: str, message: str) -> int:
            self.sent.append((channel, message))
            return 1

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p, raising=True)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")

    class _Mapping:
        def __init__(self) -> None:
            self._vals = {"request_id": "r", "user_id": 9, "model_id": "m"}

        def get(self, key: str, default: object = None) -> object:
            return self._vals.get(key, default)

    dj._emit_failed(_Mapping(), "user", "bad")

    joined = "\n".join([m for _, m in p.sent])
    assert "digits.train.failed.v1" in joined


def test_emit_interrupted_success_and_publish_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Pub:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def publish(self, channel: str, message: str) -> int:
            self.sent.append((channel, message))
            return 1

    pub = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: pub, raising=True)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    dj._emit_interrupted(
        {"request_id": "r", "user_id": 1, "model_id": "m"},
        run_id="run",
        path="/p",
    )
    joined = "\n".join([m for _, m in pub.sent])
    assert "digits.train.interrupted.v1" in joined

    class _BadPub:
        def publish(self, channel: str, message: str) -> int:
            raise OSError("fail-int")

    monkeypatch.setattr(dj, "_make_publisher", lambda: _BadPub(), raising=True)
    with pytest.raises(OSError, match="fail-int"):
        dj._emit_interrupted(
            {"request_id": "r", "user_id": 1, "model_id": "m"},
            run_id=None,
            path="/p",
        )


def test_emit_interrupted_no_publisher_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dj, "_make_publisher", lambda: None, raising=True)
    dj._emit_interrupted({"request_id": "r", "user_id": 1, "model_id": "m"}, run_id=None, path="/p")


def test_emit_interrupted_with_non_mapping_payload_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dj, "_make_publisher", lambda: None, raising=True)
    dj._emit_interrupted("not-a-dict", run_id=None, path="/p")


def test_process_train_job_completed_publish_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure earlier events publish, but completed publish fails with OSError
    class _Pub:
        def __init__(self) -> None:
            self.sent: list[tuple[str, str]] = []

        def publish(self, channel: str, message: str) -> int:
            self.sent.append((channel, message))
            return 1

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p, raising=True)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_run_training", _quick_training, raising=True)

    from handwriting_ai.events import digits as ev

    def _pub_safe(ctx: object, event: ev.EventV1) -> None:
        if isinstance(event, dict) and event.get("type") == "digits.train.completed.v1":
            raise OSError("boom-completed")
        if isinstance(ctx, dj._Context) and ctx.publisher is not None:
            ctx.publisher.publish(ctx.channel, ev.encode_event(event))

    monkeypatch.setattr(dj, "_publish_event_safe", _pub_safe, raising=True)

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r2",
        "user_id": 7,
        "model_id": "m2",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    with pytest.raises(OSError, match="boom-completed"):
        dj.process_train_job(payload)

    joined = "\n".join([m for _, m in p.sent])
    assert "digits.train.started.v1" in joined
    assert "digits.train.artifact.v1" in joined
