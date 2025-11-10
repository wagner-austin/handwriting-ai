from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.preprocess import preprocess_signature
from handwriting_ai.training.mnist_train import TrainConfig


class _BadPub:
    def publish(self, channel: str, message: str) -> int:
        raise OSError("fail")


def _quick_training(cfg: TrainConfig) -> Path:
    # Minimal dir with manifest.json to satisfy post-training steps
    root = Path(tempfile.mkdtemp())
    man: dict[str, object] = {
        "schema_version": "v1.1",
        "model_id": cfg.model_id,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.1,
        "temperature": 1.0,
    }
    (root / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return root


def test_process_train_job_publish_failures_raise_after_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: _BadPub(), raising=True)
    monkeypatch.setattr(dj, "_run_training", _quick_training, raising=True)

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 9,
        "model_id": "m1",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }

    # Should raise when publisher fails during started event
    with pytest.raises(OSError, match="fail"):
        dj.process_train_job(payload)
