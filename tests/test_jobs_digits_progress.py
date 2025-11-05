from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import handwriting_ai.jobs.digits as dj
import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.mnist_train import TrainConfig


class _Pub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.sent.append((channel, message))
        return 1


class _TinyBase:
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        for y in range(10, 18):
            for x in range(12, 16):
                img.putpixel((x, y), 255)
        return img, idx % 10


class _BadPub:
    def publish(self, channel: str, message: str) -> int:
        raise OSError("fail")


def test_process_train_job_emits_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pub = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: pub)

    # Fake training that runs real train_with_config on tiny data to drive progress
    def _realish(cfg: TrainConfig) -> Path:
        train_base = _TinyBase(4)
        test_base = _TinyBase(2)
        # ensure output dir is within tmp
        cfg2 = TrainConfig(
            data_root=tmp_path / "data",
            out_dir=tmp_path / "out",
            model_id=cfg.model_id,
            epochs=2,
            batch_size=2,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            seed=cfg.seed,
            device=cfg.device,
            optim=cfg.optim,
            scheduler="none",
            step_size=1,
            gamma=cfg.gamma,
            min_lr=cfg.min_lr,
            patience=0,
            min_delta=cfg.min_delta,
            threads=0,
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
        )
        return mt.train_with_config(cfg2, (train_base, test_base))

    monkeypatch.setattr(dj, "_run_training", _realish)

    payload: dict[str, object] = {
        "type": "digits.train.v1",
        "request_id": "r1",
        "user_id": 9,
        "model_id": "m1",
        "epochs": 2,
        "batch_size": 2,
        "lr": 0.001,
        "seed": 3,
        "augment": False,
        "notes": None,
    }

    dj.process_train_job(payload)

    msgs = [m for _, m in pub.sent]
    # Only versioned events are emitted (no legacy)
    joined = "\n".join(msgs)
    assert "digits.train.started.v1" in joined
    assert "digits.train.epoch.v1" in joined
    assert "digits.train.completed.v1" in joined
    assert "digits.train.artifact.v1" in joined


def test_process_train_job_emitters_with_bad_publisher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Use a bad publisher to exercise error paths inside the emitters
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: _BadPub())

    def _realish(cfg: TrainConfig) -> Path:
        train_base = _TinyBase(2)
        test_base = _TinyBase(1)
        cfg2 = TrainConfig(
            data_root=tmp_path / "data",
            out_dir=tmp_path / "out",
            model_id=cfg.model_id,
            epochs=1,
            batch_size=1,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            seed=cfg.seed,
            device=cfg.device,
            optim=cfg.optim,
            scheduler="none",
            step_size=1,
            gamma=cfg.gamma,
            min_lr=cfg.min_lr,
            patience=0,
            min_delta=cfg.min_delta,
            threads=0,
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
        )
        return mt.train_with_config(cfg2, (train_base, test_base))

    monkeypatch.setattr(dj, "_run_training", _realish)

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

    # Should complete without raising despite publisher failures inside emitters
    dj.process_train_job(payload)
