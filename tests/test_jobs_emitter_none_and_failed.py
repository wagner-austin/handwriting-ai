from __future__ import annotations

import pytest

import handwriting_ai.jobs.digits as dj


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
        epoch=1,
        total_epochs=1,
        batch=1,
        total_batches=1,
        batch_loss=0.1,
        batch_acc=0.9,
        avg_loss=0.1,
        samples_per_sec=100.0,
    )
    em.emit_best(epoch=1, val_acc=0.5)
    em.emit_epoch(epoch=1, total_epochs=1, train_loss=0.1, val_acc=0.2, time_s=0.1)


def test_emit_failed_without_publisher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dj, "_make_publisher", lambda: None, raising=True)
    # Should not raise when publisher is None
    dj._emit_failed({}, "user", "bad")
