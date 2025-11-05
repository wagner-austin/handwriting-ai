from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.training.mnist_train import TrainConfig


class _Pub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.sent.append((channel, message))
        return 1


class _BadPub:
    def publish(self, channel: str, message: str) -> int:
        raise OSError("fail")


def test_publish_event_error_branch_is_swallowed() -> None:
    evt: dj.DigitsTrainStartedEvent = {
        "type": "started",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 1,
    }
    # Should not raise when publisher errors
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


def test_process_train_job_invalid_payload_fields_publishes_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    dj.process_train_job(payload)
    assert any('"type":"failed"' in m for _, m in p.sent)


def test_process_train_job_training_error_publishes_failed_and_reraises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    with pytest.raises(RuntimeError):
        dj.process_train_job(payload)
    msgs = [m for _, m in p.sent]
    assert any('"type":"failed"' in m for m in msgs)


def test_emit_failed_with_non_dict_payload_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    dj._emit_failed("not-a-dict", "user", "msg")
    # One or more events may be published (legacy + v1)
    assert len(p.sent) >= 1
    ch, msg = p.sent[0]
    assert ch == "digits:events"
    compact = msg.replace(" ", "")
    assert '"type":"failed"' in compact
    assert '"request_id":""' in compact
    assert '"user_id":0' in compact
    assert '"model_id":""' in compact


def test_emit_failed_v1_publish_error_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the v1 failed publish error path is covered
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: _BadPub())
    dj._emit_failed("not-a-dict", "user", "msg")


def test_versioned_publish_errors_swallowed(
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
    # Should not raise despite publisher errors
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
