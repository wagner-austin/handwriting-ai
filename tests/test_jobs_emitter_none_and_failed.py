from __future__ import annotations

import handwriting_ai.jobs.digits as dj
from handwriting_ai.events.digits import BatchMetrics


def test_progress_emitter_no_publisher_noops() -> None:
    em = dj._ProgressEmitter(
        publisher=None,
        channel="ch",
        request_id="r",
        user_id=1,
        model_id="m",
        total_epochs=1,
    )
    em.emit_batch(
        BatchMetrics(
            epoch=1,
            total_epochs=1,
            batch=1,
            total_batches=1,
            batch_loss=0.1,
            batch_acc=0.9,
            avg_loss=0.1,
            samples_per_sec=100.0,
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
    em.emit_best(epoch=1, val_acc=0.5)
    em.emit_epoch(epoch=1, total_epochs=1, train_loss=0.1, val_acc=0.2, time_s=0.1)


# Test removed: _emit_failed no longer exists
# Failure notifications are now published exclusively by the watcher
# when it detects jobs in the failed/canceled registries
