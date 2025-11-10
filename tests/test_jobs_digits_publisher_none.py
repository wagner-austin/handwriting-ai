from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.preprocess import preprocess_signature
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


def _quick_training(cfg: TrainConfig) -> Path:
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


def test_process_train_job_with_no_publisher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: None, raising=True)

    def _quick_training_raises(cfg: TrainConfig) -> Path:
        # Raise to test that error is raised even without publisher
        raise RuntimeError("training failed")

    monkeypatch.setattr(dj, "_run_training", _quick_training_raises, raising=True)

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

    # Should raise even without publisher
    with pytest.raises(RuntimeError, match="training failed"):
        dj.process_train_job(payload)
