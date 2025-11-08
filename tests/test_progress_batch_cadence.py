from __future__ import annotations

from typing import TypedDict

import handwriting_ai.training.progress as prog


class _BatchMsg(TypedDict):
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float


class _Collector:
    def __init__(self) -> None:
        self.items: list[_BatchMsg] = []

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
        self.items.append(
            {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch": batch,
                "total_batches": total_batches,
                "batch_loss": batch_loss,
                "batch_acc": batch_acc,
                "avg_loss": avg_loss,
                "samples_per_sec": samples_per_sec,
            }
        )


def test_batch_cadence_emits_first_multiples_and_last() -> None:
    col = _Collector()
    prog.set_batch_emitter(col)
    prog.set_batch_cadence(100)
    total_batches = 205
    try:
        for i in range(1, total_batches + 1):
            prog.emit_batch(
                epoch=1,
                total_epochs=15,
                batch=i,
                total_batches=total_batches,
                batch_loss=0.1,
                batch_acc=0.9,
                avg_loss=0.2,
                samples_per_sec=50.0,
            )
    finally:
        # Reset cadence to avoid cross-test leakage
        prog.set_batch_cadence(0)
    seen = [m["batch"] for m in col.items]
    assert seen == [1, 100, 200, 205]
