from __future__ import annotations

import json

from handwriting_ai.events import digits as ev


def test_encode_all_event_builders_cover_fields() -> None:
    ctx = ev.Context(request_id="r", user_id=7, model_id="m", run_id=None)

    s = ev.started(
        ctx,
        total_epochs=2,
        cpu_cores=2,
        memory_mb=953,
        optimal_threads=2,
        optimal_workers=0,
        max_batch_size=64,
        device="cpu",
        batch_size=32,
        augment=False,
        aug_rotate=10.0,
        aug_translate=0.1,
        noise_prob=0.0,
        dots_prob=0.0,
    )
    assert s["type"] == "digits.train.started.v1"
    assert s["total_epochs"] == 2

    e = ev.epoch(ctx, epoch=1, total_epochs=2, train_loss=0.1, val_acc=0.2, time_s=1.5)
    assert e["type"] == "digits.train.epoch.v1"
    assert e["epoch"] == 1

    b = ev.batch(
        ctx,
        epoch=1,
        total_epochs=2,
        batch=3,
        total_batches=10,
        batch_loss=0.5,
        batch_acc=0.9,
        avg_loss=0.6,
        samples_per_sec=100.0,
    )
    assert b["type"] == "digits.train.batch.v1"
    assert b["batch"] == 3

    be = ev.best(ctx, epoch=1, val_acc=0.3)
    assert be["type"] == "digits.train.best.v1"
    assert be["val_acc"] == 0.3

    a = ev.artifact(ctx, path="/x/y")
    assert a["type"] == "digits.train.artifact.v1"
    assert a["path"] == "/x/y"

    u = ev.upload(ctx, status=200, model_bytes=11, manifest_bytes=7)
    assert u["type"] == "digits.train.upload.v1"
    assert u["model_bytes"] == 11 and u["manifest_bytes"] == 7

    p = ev.prune(ctx, deleted_count=4)
    assert p["type"] == "digits.train.prune.v1"
    assert p["deleted_count"] == 4

    c = ev.completed(ctx, val_acc=0.99)
    assert c["type"] == "digits.train.completed.v1"
    assert c["val_acc"] == 0.99

    f = ev.failed(ctx, error_kind="user", message="bad")
    assert f["type"] == "digits.train.failed.v1"
    assert f["message"] == "bad"

    # Encode coverage
    s_json = ev.encode_event(s)
    obj: dict[str, object] = json.loads(s_json)
    assert obj["type"] == "digits.train.started.v1"
