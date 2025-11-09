from __future__ import annotations

from typing import TypedDict

import handwriting_ai.training.progress as prog
from handwriting_ai.events.digits import BatchMetrics


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

    def emit_batch(self, metrics: BatchMetrics) -> None:
        self.items.append(
            {
                "epoch": metrics.epoch,
                "total_epochs": metrics.total_epochs,
                "batch": metrics.batch,
                "total_batches": metrics.total_batches,
                "batch_loss": metrics.batch_loss,
                "batch_acc": metrics.batch_acc,
                "avg_loss": metrics.avg_loss,
                "samples_per_sec": metrics.samples_per_sec,
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
                BatchMetrics(
                    epoch=1,
                    total_epochs=15,
                    batch=i,
                    total_batches=total_batches,
                    batch_loss=0.1,
                    batch_acc=0.9,
                    avg_loss=0.2,
                    samples_per_sec=50.0,
                    main_rss_mb=100,
                    workers_rss_mb=50,
                    worker_count=2,
                    cgroup_usage_mb=500,
                    cgroup_limit_mb=1000,
                    cgroup_pct=50.0,
                    anon_mb=200,
                    file_mb=150,
                )
            )
    finally:
        # Reset cadence to avoid cross-test leakage
        prog.set_batch_cadence(0)
    seen = [m["batch"] for m in col.items]
    assert seen == [1, 100, 200, 205]
