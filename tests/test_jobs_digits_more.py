from __future__ import annotations

import json
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.training.mnist_train import TrainConfig
from handwriting_ai.training.resources import ResourceLimits


@pytest.fixture(autouse=True)
def _mock_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock resource detection for Windows/non-container environments."""
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=4 * 1024 * 1024 * 1024,
        optimal_threads=2,
        optimal_workers=0,
        max_batch_size=64,
    )
    import handwriting_ai.training.runtime as rt

    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits, raising=False)


class _Pub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.sent.append((channel, message))
        return 1


class _BadPub:
    def publish(self, channel: str, message: str) -> int:
        raise OSError("fail")


def test_publish_event_error_raises_after_logging() -> None:
    evt: dj.DigitsTrainStartedEvent = {
        "type": "started",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
    }
    # Should raise after logging when publisher errors
    with pytest.raises(OSError, match="fail"):
        dj._publish_event(_BadPub(), "digits:events", evt)


def test_publish_event_no_publisher_noop() -> None:
    evt: dj.DigitsTrainStartedEvent = {
        "type": "started",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
    }
    # Should simply return when publisher is None
    dj._publish_event(None, "digits:events", evt)


def test_process_train_job_invalid_payload_fields_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Payload validation errors now propagate to RQ's default handler
    # Failed events are published by watcher, not by process_train_job
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    # Wrong types to trigger ValueError in coercion
    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": True,  # bool is explicitly rejected in _as_int
        "model_id": "m1",
        "epochs": "one",
        "batch_size": 4,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }
    # Should raise ValueError after logging
    with pytest.raises(ValueError):
        dj.process_train_job(payload)


def test_process_train_job_training_error_reraises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Training errors now propagate to RQ's default handler
    # Failed events are published by watcher, not by process_train_job
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)

    def _raise_run(_: TrainConfig) -> Path:
        raise RuntimeError("boom")

    monkeypatch.setattr(dj, "_run_training", _raise_run)
    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 2,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }
    with pytest.raises(RuntimeError, match="boom"):
        dj.process_train_job(payload)
    # Verify started event was published before failure
    joined = "\n".join([m for _, m in p.sent])
    assert "digits.train.started.v1" in joined


# Tests removed: _emit_failed no longer exists
# Failure notifications are now published exclusively by the watcher
# when it detects jobs in the failed/canceled registries


def test_versioned_publish_errors_raise_after_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Use a bad publisher to trigger the try/except branches around v1 event publishing
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: _BadPub())

    def _fake_run(cfg: TrainConfig) -> Path:
        d = tmp_path / cfg.model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"pt")
        manifest = {
            "schema_version": "v1.1",
            "model_id": cfg.model_id,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": "2025-01-01T00:00:00+00:00",
            "preprocess_hash": "h",
            "val_acc": 0.9,
            "temperature": 1.0,
        }
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return d

    import json

    monkeypatch.setattr(dj, "_run_training", _fake_run)
    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 1,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }
    # Should raise when publisher errors during started event
    with pytest.raises(OSError, match="fail"):
        dj.process_train_job(payload)


def test_run_training_default_raises(tmp_path: Path) -> None:
    cfg = TrainConfig(
        data_root=tmp_path / "data",
        out_dir=tmp_path / "out",
        model_id="m",
        epochs=1,
        batch_size=1,
        lr=0.001,
        weight_decay=1e-2,
        seed=0,
        device="cpu",
        optim="adamw",
        scheduler="cosine",
        step_size=10,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=10.0,
        aug_translate=0.1,
    )
    with pytest.raises(RuntimeError):
        dj._run_training(cfg)


def test_make_publisher_default_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIGITS_EVENTS_CHANNEL", raising=False)
    pub = dj._make_publisher()
    assert pub is None
    ctx = dj._make_context()
    assert ctx.publisher is None
    assert ctx.channel == dj.DEFAULT_EVENTS_CHANNEL


def test_as_int_and_as_float_edges() -> None:
    # _as_int
    assert dj._as_int(7) == 7
    assert dj._as_int("8") == 8
    with pytest.raises(ValueError):
        dj._as_int(True)
    with pytest.raises(ValueError):
        dj._as_int(object())
    # _as_float
    assert dj._as_float(1.5) == 1.5
    assert dj._as_float(3) == 3.0
    assert dj._as_float("4.5") == 4.5
    with pytest.raises(ValueError):
        dj._as_float(False)
    with pytest.raises(ValueError):
        dj._as_float(object())


def test_summarize_training_exception_memory_pressure_guard() -> None:
    """Test _summarize_training_exception formats memory_pressure_guard_triggered with threshold."""
    exc = RuntimeError("memory_pressure_guard_triggered")
    result = dj._summarize_training_exception(exc)
    # Verify message structure and content
    assert "Training aborted due to sustained memory pressure" in result
    assert "Reduce batch size or DataLoader workers and retry" in result
    # Verify threshold percentage is included
    assert ">=" in result
    assert "%" in result
    # Verify it's a user-friendly message, not a raw exception
    assert "RuntimeError" not in result
    assert "memory_pressure_guard_triggered" not in result


def test_summarize_training_exception_artifact_upload_failed() -> None:
    """Test _summarize_training_exception recognizes artifact upload failures."""
    exc = RuntimeError("artifact upload failed: connection timeout")
    result = dj._summarize_training_exception(exc)
    # Verify exact expected message
    assert result == "Artifact upload failed: upstream API error. See worker logs for details."
    # Verify original error details are not leaked to user
    assert "connection timeout" not in result


def test_summarize_training_exception_generic() -> None:
    """Test _summarize_training_exception handles generic exceptions."""
    exc = ValueError("invalid configuration parameter")
    result = dj._summarize_training_exception(exc)
    # Verify exception type and message are included
    assert result == "ValueError: invalid configuration parameter"


def test_summarize_training_exception_long_message_truncated() -> None:
    """Test _summarize_training_exception truncates messages longer than 300 chars."""
    long_msg = "x" * 400
    exc = RuntimeError(long_msg)
    result = dj._summarize_training_exception(exc)
    # Verify message is truncated to 300 chars (plus exception type prefix)
    assert result.startswith("RuntimeError: ")
    assert len(result) <= len("RuntimeError: ") + 300
    assert result.endswith("x")
    # Verify the preview is exactly 300 chars of the message
    preview_part = result[len("RuntimeError: ") :]
    assert len(preview_part) == 300


def test_summarize_training_exception_empty_message() -> None:
    """Test _summarize_training_exception handles exceptions with no message."""
    exc = ValueError()
    result = dj._summarize_training_exception(exc)
    # Verify only exception type is returned when message is empty
    assert result == "ValueError"


def test_process_train_job_queue_name_from_rq_job_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test queue name is extracted from RQ job context and included in started event."""

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")

    def _fake_run(cfg: TrainConfig) -> Path:
        d = tmp_path / cfg.model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"pt")
        manifest = {
            "schema_version": "v1.1",
            "model_id": cfg.model_id,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": "2025-01-01T00:00:00+00:00",
            "preprocess_hash": "h",
            "val_acc": 0.9,
            "temperature": 1.0,
        }
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return d

    monkeypatch.setattr(dj, "_run_training", _fake_run)

    # Mock get_current_job to return a job with origin
    class _MockJob:
        origin: str | None = "digits-priority"

    monkeypatch.setattr(dj, "get_current_job", lambda: _MockJob())

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r-queue",
        "user_id": 1,
        "model_id": "m-queue",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    dj.process_train_job(payload)

    # Find and parse started event
    started_event: dict[str, object] | None = None
    for _, msg in p.sent:
        data: dict[str, object] = json.loads(msg)
        evt_type: object = data.get("type")
        if evt_type == "digits.train.started.v1":
            started_event = data
            break

    assert started_event is not None, "Started event not found"
    # Deeply inspect: verify queue name is exactly as provided by RQ
    assert started_event["queue"] == "digits-priority"
    assert isinstance(started_event["queue"], str)
    # Verify other required fields are present
    assert started_event["request_id"] == "r-queue"
    assert started_event["model_id"] == "m-queue"


def test_process_train_job_queue_name_none_when_not_in_rq_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test queue name is None when get_current_job() returns None."""

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")

    def _fake_run(cfg: TrainConfig) -> Path:
        d = tmp_path / cfg.model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"pt")
        manifest = {
            "schema_version": "v1.1",
            "model_id": cfg.model_id,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": "2025-01-01T00:00:00+00:00",
            "preprocess_hash": "h",
            "val_acc": 0.9,
            "temperature": 1.0,
        }
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return d

    monkeypatch.setattr(dj, "_run_training", _fake_run)

    # Mock get_current_job to return None (not in RQ context)
    monkeypatch.setattr(dj, "get_current_job", lambda: None)

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r-no-rq",
        "user_id": 1,
        "model_id": "m-no-rq",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    dj.process_train_job(payload)

    # Find and parse started event
    started_event: dict[str, object] | None = None
    for _, msg in p.sent:
        data: dict[str, object] = json.loads(msg)
        evt_type: object = data.get("type")
        if evt_type == "digits.train.started.v1":
            started_event = data
            break

    assert started_event is not None, "Started event not found"
    # Deeply inspect: verify queue is explicitly None, not a string or missing
    assert started_event["queue"] is None
    assert "queue" in started_event  # Key must exist
    # Verify other required fields are present
    assert started_event["request_id"] == "r-no-rq"
    assert started_event["model_id"] == "m-no-rq"


def test_process_train_job_queue_name_none_when_job_origin_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test queue name is None when RQ job exists but origin is None."""

    p = _Pub()
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")

    def _fake_run(cfg: TrainConfig) -> Path:
        d = tmp_path / cfg.model_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.pt").write_bytes(b"pt")
        manifest = {
            "schema_version": "v1.1",
            "model_id": cfg.model_id,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": "2025-01-01T00:00:00+00:00",
            "preprocess_hash": "h",
            "val_acc": 0.9,
            "temperature": 1.0,
        }
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return d

    monkeypatch.setattr(dj, "_run_training", _fake_run)

    # Mock get_current_job to return a job with origin=None
    class _MockJob:
        origin: str | None = None

    monkeypatch.setattr(dj, "get_current_job", lambda: _MockJob())

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r-no-origin",
        "user_id": 1,
        "model_id": "m-no-origin",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 1,
        "augment": False,
        "notes": None,
    }

    dj.process_train_job(payload)

    # Find and parse started event
    started_event: dict[str, object] | None = None
    for _, msg in p.sent:
        data: dict[str, object] = json.loads(msg)
        evt_type: object = data.get("type")
        if evt_type == "digits.train.started.v1":
            started_event = data
            break

    assert started_event is not None, "Started event not found"
    # Deeply inspect: verify queue is explicitly None when origin is None
    assert started_event["queue"] is None
    assert "queue" in started_event  # Key must exist
    # Verify other required fields are present
    assert started_event["request_id"] == "r-no-origin"
    assert started_event["model_id"] == "m-no-origin"
