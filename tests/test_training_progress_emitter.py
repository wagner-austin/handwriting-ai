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
from handwriting_ai.training.progress import (
    emit_batch,
    emit_best,
    emit_epoch,
    set_batch_emitter,
    set_best_emitter,
    set_epoch_emitter,
)


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


def test_progress_emitter_every_n_epochs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    # Increase epochs and set cadence
    cfg = TrainConfig(
        data_root=cfg.data_root,
        out_dir=cfg.out_dir,
        model_id=cfg.model_id,
        epochs=5,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
        device=cfg.device,
        optim=cfg.optim,
        scheduler=cfg.scheduler,
        step_size=cfg.step_size,
        gamma=cfg.gamma,
        min_lr=cfg.min_lr,
        patience=0,
        min_delta=cfg.min_delta,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        progress_every_epochs=2,
    )
    train_base = _TinyBase(6)
    test_base = _TinyBase(3)

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

    rec = _Rec()
    monkeypatch.setattr(mt, "_train_epoch", _ok_train_epoch, raising=True)
    set_progress_emitter(rec)
    try:
        out = mt.train_with_config(cfg, (train_base, test_base))
        assert (out / "model.pt").exists()
    finally:
        set_progress_emitter(None)


class _BatchBestEpochRecorder:
    def __init__(self) -> None:
        self.batch: list[tuple[int, int, int]] = []
        self.best: list[tuple[int, float]] = []
        self.epoch: list[tuple[int, float, float]] = []

    def emit_batch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        batch: int,
        total_batches: int,
        batch_loss: float,
        batch_acc: float,
        avg_loss: float,
        samples_per_sec: float,
    ) -> None:
        self.batch.append((epoch, total_epochs, batch))

    def emit_best(self, *, epoch: int, val_acc: float) -> None:
        self.best.append((epoch, val_acc))

    def emit_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_acc: float,
        time_s: float,
    ) -> None:
        self.epoch.append((epoch, train_loss, val_acc))


def test_batch_best_epoch_emitters_are_called() -> None:
    rec = _BatchBestEpochRecorder()
    set_batch_emitter(rec)
    set_best_emitter(rec)
    set_epoch_emitter(rec)

    emit_batch(
        epoch=1,
        total_epochs=2,
        batch=3,
        total_batches=10,
        batch_loss=0.1,
        batch_acc=0.9,
        avg_loss=0.2,
        samples_per_sec=50.0,
    )
    emit_best(epoch=1, val_acc=0.5)
    emit_epoch(epoch=1, total_epochs=2, train_loss=0.3, val_acc=0.4, time_s=1.0)

    assert len(rec.batch) > 0 and len(rec.best) > 0 and len(rec.epoch) > 0


def test_batch_best_epoch_emitter_failures_swallowed() -> None:
    class _Bad:
        def emit_batch(
            self,
            *,
            epoch: int,
            total_epochs: int,
            batch: int,
            total_batches: int,
            batch_loss: float,
            batch_acc: float,
            avg_loss: float,
            samples_per_sec: float,
        ) -> None:
            raise ValueError("boom")

        def emit_best(self, *, epoch: int, val_acc: float) -> None:
            raise ValueError("boom")

        def emit_epoch(
            self,
            *,
            epoch: int,
            total_epochs: int,
            train_loss: float,
            val_acc: float,
            time_s: float,
        ) -> None:
            raise ValueError("boom")

    bad = _Bad()
    set_batch_emitter(bad)
    set_best_emitter(bad)
    set_epoch_emitter(bad)

    # Calls should not raise
    emit_batch(
        epoch=1,
        total_epochs=2,
        batch=1,
        total_batches=2,
        batch_loss=0.1,
        batch_acc=0.9,
        avg_loss=0.2,
        samples_per_sec=10.0,
    )
    emit_best(epoch=1, val_acc=0.8)
    emit_epoch(epoch=1, total_epochs=2, train_loss=0.3, val_acc=0.4, time_s=1.0)


def test_emit_no_emitters_noop() -> None:
    # Ensure calling emitters without setting them is a no-op (covers early returns)
    set_batch_emitter(None)
    set_best_emitter(None)
    set_epoch_emitter(None)
    emit_batch(
        epoch=1,
        total_epochs=2,
        batch=1,
        total_batches=2,
        batch_loss=0.1,
        batch_acc=0.9,
        avg_loss=0.2,
        samples_per_sec=10.0,
    )
    emit_best(epoch=1, val_acc=0.7)
    emit_epoch(epoch=1, total_epochs=2, train_loss=0.1, val_acc=0.2, time_s=0.5)
