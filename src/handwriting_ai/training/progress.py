from __future__ import annotations

import logging
from typing import Protocol


class ProgressEmitter(Protocol):
    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None: ...


_emitter: ProgressEmitter | None = None


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
