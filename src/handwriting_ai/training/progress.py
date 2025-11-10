from __future__ import annotations

import logging
from typing import Protocol

from handwriting_ai.events.digits import BatchMetrics


class ProgressEmitter(Protocol):
    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None: ...


_emitter: ProgressEmitter | None = None
_batch_emitter: BatchProgressEmitter | None = None
_best_emitter: BestEmitter | None = None
_batch_cadence: int = 0  # 0 means emit all batches


def set_progress_emitter(emitter: ProgressEmitter | None) -> None:
    global _emitter
    _emitter = emitter


def emit_progress(epoch: int, total_epochs: int, val_acc: float | None) -> None:
    em = _emitter
    if em is None:
        return
    try:
        em.emit(epoch=epoch, total_epochs=total_epochs, val_acc=val_acc)
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger("handwriting_ai").error("progress_emitter_failed error=%s", exc)
        raise


class BatchProgressEmitter(Protocol):
    """Protocol for batch progress emitters.

    Single source of truth: accepts BatchMetrics dataclass.
    """

    def emit_batch(self, metrics: BatchMetrics) -> None: ...


def set_batch_emitter(emitter: BatchProgressEmitter | None) -> None:
    global _batch_emitter
    _batch_emitter = emitter


def set_batch_cadence(cadence: int) -> None:
    """Configure global batch progress emission cadence.

    - cadence <= 0: emit every batch
    - cadence > 0: emit batches where (batch % cadence == 0), plus first and last batch
    """
    global _batch_cadence
    _batch_cadence = int(cadence)


def emit_batch(metrics: BatchMetrics) -> None:
    """Emit batch progress using global emitter.

    Single source of truth: accepts BatchMetrics dataclass.
    """
    em = _batch_emitter
    if em is None:
        return
    # Gate emission by global cadence to centralize frequency control
    cad = int(_batch_cadence)
    if cad > 0 and not (
        metrics.batch == 1 or metrics.batch == metrics.total_batches or (metrics.batch % cad == 0)
    ):
        return
    try:
        em.emit_batch(metrics)
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger("handwriting_ai").error("progress_batch_emitter_failed error=%s", exc)
        raise


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
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger("handwriting_ai").error("progress_best_emitter_failed error=%s", exc)
        raise


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
    except (RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger("handwriting_ai").error("progress_epoch_emitter_failed error=%s", exc)
        raise
