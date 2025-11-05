from __future__ import annotations

import logging
from typing import Protocol


class ProgressEmitter(Protocol):
    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None: ...


_emitter: ProgressEmitter | None = None
_batch_emitter: BatchProgressEmitter | None = None
_best_emitter: BestEmitter | None = None


def set_progress_emitter(emitter: ProgressEmitter | None) -> None:
    global _emitter
    _emitter = emitter


def emit_progress(epoch: int, total_epochs: int, val_acc: float | None) -> None:
    em = _emitter
    if em is None:
        return
    try:
        em.emit(epoch=epoch, total_epochs=total_epochs, val_acc=val_acc)
    except (RuntimeError, ValueError, TypeError):
        logging.getLogger("handwriting_ai").debug("progress_emitter_failed")


class BatchProgressEmitter(Protocol):
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
    ) -> None: ...


def set_batch_emitter(emitter: BatchProgressEmitter | None) -> None:
    global _batch_emitter
    _batch_emitter = emitter


def emit_batch(
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
    em = _batch_emitter
    if em is None:
        return
    try:
        em.emit_batch(
            epoch=epoch,
            total_epochs=total_epochs,
            batch=batch,
            total_batches=total_batches,
            batch_loss=batch_loss,
            batch_acc=batch_acc,
            avg_loss=avg_loss,
            samples_per_sec=samples_per_sec,
        )
    except (RuntimeError, ValueError, TypeError):
        logging.getLogger("handwriting_ai").debug("progress_batch_emitter_failed")


class BestEmitter(Protocol):
    def emit_best(self, *, epoch: int, val_acc: float) -> None: ...


def set_best_emitter(emitter: BestEmitter | None) -> None:
    global _best_emitter
    _best_emitter = emitter


def emit_best(*, epoch: int, val_acc: float) -> None:
    em = _best_emitter
    if em is None:
        return
    try:
        em.emit_best(epoch=epoch, val_acc=val_acc)
    except (RuntimeError, ValueError, TypeError):
        logging.getLogger("handwriting_ai").debug("progress_best_emitter_failed")


class EpochEmitter(Protocol):
    def emit_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_acc: float,
        time_s: float,
    ) -> None: ...


_epoch_emitter: EpochEmitter | None = None


def set_epoch_emitter(emitter: EpochEmitter | None) -> None:
    global _epoch_emitter
    _epoch_emitter = emitter


def emit_epoch(
    *, epoch: int, total_epochs: int, train_loss: float, val_acc: float, time_s: float
) -> None:
    em = _epoch_emitter
    if em is None:
        return
    try:
        em.emit_epoch(
            epoch=epoch,
            total_epochs=total_epochs,
            train_loss=train_loss,
            val_acc=val_acc,
            time_s=time_s,
        )
    except (RuntimeError, ValueError, TypeError):
        logging.getLogger("handwriting_ai").debug("progress_epoch_emitter_failed")
