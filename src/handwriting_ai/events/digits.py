from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict


@dataclass(frozen=True)
class BatchMetrics:
    """Single source of truth for batch progress metrics.

    This dataclass defines all batch-level metrics exactly once.
    All other schemas, protocols, and function signatures derive from this.
    """

    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    batch_loss: float
    batch_acc: float
    avg_loss: float
    samples_per_sec: float
    # Memory metrics (from cgroup-aware monitoring)
    main_rss_mb: int
    workers_rss_mb: int
    worker_count: int
    cgroup_usage_mb: int
    cgroup_limit_mb: int
    cgroup_pct: float
    anon_mb: int
    file_mb: int


class StartedV1(TypedDict):
    type: Literal["digits.train.started.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    total_epochs: int
    queue: str | None
    cpu_cores: int
    memory_mb: int | None
    optimal_threads: int
    optimal_workers: int
    max_batch_size: int | None
    device: str
    batch_size: int
    learning_rate: float
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    dots_prob: float


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
    # Memory metrics (from cgroup-aware monitoring)
    main_rss_mb: int
    workers_rss_mb: int
    worker_count: int
    cgroup_usage_mb: int
    cgroup_limit_mb: int
    cgroup_pct: float
    anon_mb: int
    file_mb: int


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
    queue: str | None
    status: Literal["failed", "canceled"]


class InterruptedV1(TypedDict):
    type: Literal["digits.train.interrupted.v1"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str | None
    ts: str
    path: str


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
    | InterruptedV1
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


def started(
    ctx: Context,
    *,
    total_epochs: int,
    queue: str | None,
    cpu_cores: int,
    memory_mb: int | None,
    optimal_threads: int,
    optimal_workers: int,
    max_batch_size: int | None,
    device: str,
    batch_size: int,
    learning_rate: float,
    augment: bool,
    aug_rotate: float,
    aug_translate: float,
    noise_prob: float,
    dots_prob: float,
) -> StartedV1:
    return {
        "type": "digits.train.started.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "total_epochs": int(total_epochs),
        "queue": (str(queue) if queue is not None else None),
        "cpu_cores": int(cpu_cores),
        "memory_mb": (int(memory_mb) if isinstance(memory_mb, int) else None),
        "optimal_threads": int(optimal_threads),
        "optimal_workers": int(optimal_workers),
        "max_batch_size": (int(max_batch_size) if max_batch_size is not None else None),
        "device": str(device),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "augment": bool(augment),
        "aug_rotate": float(aug_rotate),
        "aug_translate": float(aug_translate),
        "noise_prob": float(noise_prob),
        "dots_prob": float(dots_prob),
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


def batch(ctx: Context, metrics: BatchMetrics) -> BatchV1:
    """Build batch event from metrics dataclass.

    Single source of truth: BatchMetrics dataclass defines all fields.
    """
    return {
        "type": "digits.train.batch.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        # Extract all batch metrics from dataclass with proper type conversion
        "epoch": int(metrics.epoch),
        "total_epochs": int(metrics.total_epochs),
        "batch": int(metrics.batch),
        "total_batches": int(metrics.total_batches),
        "batch_loss": float(metrics.batch_loss),
        "batch_acc": float(metrics.batch_acc),
        "avg_loss": float(metrics.avg_loss),
        "samples_per_sec": float(metrics.samples_per_sec),
        "main_rss_mb": int(metrics.main_rss_mb),
        "workers_rss_mb": int(metrics.workers_rss_mb),
        "worker_count": int(metrics.worker_count),
        "cgroup_usage_mb": int(metrics.cgroup_usage_mb),
        "cgroup_limit_mb": int(metrics.cgroup_limit_mb),
        "cgroup_pct": float(metrics.cgroup_pct),
        "anon_mb": int(metrics.anon_mb),
        "file_mb": int(metrics.file_mb),
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


def failed(
    ctx: Context,
    *,
    error_kind: Literal["user", "system"],
    message: str,
    queue: str | None,
    status: Literal["failed", "canceled"],
) -> FailedV1:
    return {
        "type": "digits.train.failed.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "error_kind": error_kind,
        "message": message,
        "queue": (str(queue) if queue is not None else None),
        "status": status,
    }


def interrupted(ctx: Context, *, path: str) -> InterruptedV1:
    return {
        "type": "digits.train.interrupted.v1",
        "request_id": ctx.request_id,
        "user_id": int(ctx.user_id),
        "model_id": ctx.model_id,
        "run_id": ctx.run_id,
        "ts": _ts(),
        "path": path,
    }
