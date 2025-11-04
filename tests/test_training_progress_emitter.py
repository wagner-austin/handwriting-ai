from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.mnist_train import TrainConfig, set_progress_emitter


class _TinyBase(Dataset[tuple[Image.Image, int]]):
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


class _Rec:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float | None]] = []

    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None:
        acc: float | None = float(val_acc) if isinstance(val_acc, float) else None
        self.calls.append((int(epoch), int(total_epochs), acc))


def _cfg(tmp: Path) -> TrainConfig:
    return TrainConfig(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-2,
        seed=123,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_progress_emitter_receives_epoch_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    rec = _Rec()

    # Ensure no stray KeyboardInterrupt from patched tests
    def _ok_train_epoch(
        model: mt.TrainableModel,
        train_loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
        optimizer: object,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        return 0.0

    monkeypatch.setattr(mt, "_train_epoch", _ok_train_epoch, raising=True)
    set_progress_emitter(rec)
    try:
        out = mt.train_with_config(cfg, (train_base, test_base))
        assert (out / "model.pt").exists()
    finally:
        set_progress_emitter(None)
    # Expect an emit per epoch
    assert len(rec.calls) >= 2
    # Verify epoch numbers are within range and total matches config
    tot = rec.calls[-1][1]
    assert tot == cfg.epochs
    assert rec.calls[0][0] == 1
    assert rec.calls[-1][0] <= cfg.epochs


def test_progress_emitter_failure_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(2)
    test_base = _TinyBase(2)

    def _ok_train_epoch(
        model: mt.TrainableModel,
        train_loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
        optimizer: object,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        return 0.0

    class _Bad:
        def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None:
            raise ValueError("boom")

    monkeypatch.setattr(mt, "_train_epoch", _ok_train_epoch, raising=True)
    set_progress_emitter(_Bad())
    try:
        out = mt.train_with_config(cfg, (train_base, test_base))
        assert (out / "model.pt").exists()
    finally:
        set_progress_emitter(None)
