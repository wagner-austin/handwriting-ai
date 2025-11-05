from __future__ import annotations

import json
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


def test_encode_event_compact_json() -> None:
    evt: dj.DigitsTrainStartedEvent = {
        "type": "started",
        "request_id": "r",
        "user_id": 1,
        "model_id": "m",
        "total_epochs": 5,
    }
    s = dj.encode_event(evt)
    assert s == json.dumps(evt, separators=(",", ":"))


def test_process_train_job_invalid_type_publishes_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setenv("REDIS_URL", "")  # force None publisher path
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)
    payload: dict[str, object] = {"type": "wrong"}
    dj.process_train_job(payload)
    assert any("failed" in m for _, m in p.sent)


def test_process_train_job_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)

    # Ensure job builder resolves paths under tmp, not /data, by monkeypatching settings provider
    from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings

    def _load_settings() -> Settings:
        app = AppConfig(
            data_root=tmp_path / "data",
            artifacts_root=tmp_path / "artifacts",
            logs_root=tmp_path / "logs",
            threads=0,
            port=8081,
        )
        return Settings(app=app, digits=DigitsConfig(), security=SecurityConfig())

    monkeypatch.setattr(dj, "_load_settings", staticmethod(_load_settings))

    def _fake_run(cfg: TrainConfig) -> Path:  # create tiny artifact dir
        d = cfg.out_dir / cfg.model_id
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
    dj.process_train_job(payload)
    # Should publish v1 started and completed events
    msgs = [m for _, m in p.sent]
    joined = "\n".join(msgs)
    assert "digits.train.started.v1" in joined
    assert "digits.train.completed.v1" in joined
