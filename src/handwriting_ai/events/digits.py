from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict


class StartedV1(TypedDict):
    type: Literal["digits.train.started.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    total_epochs: int


class BatchV1(TypedDict):
    type: Literal["digits.train.batch.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float


class EpochV1(TypedDict):
    type: Literal["digits.train.epoch.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_acc: float
    time_s: float


class BestV1(TypedDict):
    type: Literal["digits.train.best.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    epoch: int
    val_acc: float


class ArtifactV1(TypedDict):
    type: Literal["digits.train.artifact.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    path: str


class UploadV1(TypedDict):
    type: Literal["digits.train.upload.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    status: int
    model_bytes: int
    manifest_bytes: int


class PruneV1(TypedDict):
    type: Literal["digits.train.prune.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    deleted_count: int


class CompletedV1(TypedDict):
    type: Literal["digits.train.completed.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    val_acc: float


class FailedV1(TypedDict):
    type: Literal["digits.train.failed.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    error_kind: Literal["user", "system"]
    message: str


EventV1 = (
    StartedV1
    | BatchV1
    | EpochV1
    | BestV1
    | ArtifactV1
    | UploadV1
    | PruneV1
    | CompletedV1
    | FailedV1
)


def encode_event(ev: EventV1) -> str:
    return json.dumps(ev, separators=(",", ":"))


def _ts() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Context:
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None


def started(ctx: Context, *, total_epochs: int) -> StartedV1:
    return {
        "type": "digits.train.started.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "total_epochs": int(total_epochs),
    }


def epoch(
    ctx: Context,
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_acc: float,
    time_s: float,
) -> EpochV1:
    return {
        "type": "digits.train.epoch.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "train_loss": float(train_loss),
        "val_acc": float(val_acc),
        "time_s": float(time_s),
    }


def batch(
    ctx: Context,
    *,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    batch_loss: float,
    batch_acc: float,
    avg_loss: float,
    samples_per_sec: float,
) -> BatchV1:
    return {
        "type": "digits.train.batch.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "epoch": int(epoch),
        "total_epochs": int(total_epochs),
        "batch": int(batch),
        "total_batches": int(total_batches),
        "batch_loss": float(batch_loss),
        "batch_acc": float(batch_acc),
        "avg_loss": float(avg_loss),
        "samples_per_sec": float(samples_per_sec),
    }


def best(ctx: Context, *, epoch: int, val_acc: float) -> BestV1:
    return {
        "type": "digits.train.best.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "epoch": int(epoch),
        "val_acc": float(val_acc),
    }


def artifact(ctx: Context, *, path: str) -> ArtifactV1:
    return {
        "type": "digits.train.artifact.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "path": path,
    }


def upload(ctx: Context, *, status: int, model_bytes: int, manifest_bytes: int) -> UploadV1:
    return {
        "type": "digits.train.upload.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "status": int(status),
        "model_bytes": int(model_bytes),
        "manifest_bytes": int(manifest_bytes),
    }


def prune(ctx: Context, *, deleted_count: int) -> PruneV1:
    return {
        "type": "digits.train.prune.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "deleted_count": int(deleted_count),
    }


def completed(ctx: Context, *, val_acc: float) -> CompletedV1:
    return {
        "type": "digits.train.completed.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "val_acc": float(val_acc),
    }


def failed(ctx: Context, *, error_kind: Literal["user", "system"], message: str) -> FailedV1:
    return {
        "type": "digits.train.failed.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "error_kind": error_kind,
        "message": message,
    }
