from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, TypedDict

from handwriting_ai.config import Settings
from handwriting_ai.events import digits as ev
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.training.mnist_train import TrainConfig, set_progress_emitter
from handwriting_ai.training.progress import (
    set_batch_emitter as _set_batch_emitter,
)
from handwriting_ai.training.progress import (
    set_best_emitter as _set_best_emitter,
)
from handwriting_ai.training.progress import (
    set_epoch_emitter as _set_epoch_emitter,
)
from handwriting_ai.training.resources import detect_resource_limits
from handwriting_ai.training.safety import get_memory_guard_config as _get_mg_cfg

DEFAULT_EVENTS_CHANNEL: Final[str] = "digits:events"


class DigitsTrainJobV1(TypedDict):
    type: Literal["digits.train.v1"]
    request_id: str
    user_id: int
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    augment: bool
    notes: str | None


class DigitsTrainStartedEvent(TypedDict):
    type: Literal["started"]
    request_id: str
    user_id: int
    model_id: str
    total_epochs: int


class DigitsTrainCompletedEvent(TypedDict):
    type: Literal["completed"]
    request_id: str
    user_id: int
    model_id: str
    run_id: str
    val_acc: float


class DigitsTrainFailedEvent(TypedDict):
    type: Literal["failed"]
    request_id: str
    user_id: int
    model_id: str
    error_kind: Literal["user", "system"]
    message: str


class DigitsTrainProgressEvent(TypedDict):
    type: Literal["progress"]
    request_id: str
    user_id: int
    model_id: str
    epoch: int
    total_epochs: int
    val_acc: float | None


Event = (
    DigitsTrainStartedEvent
    | DigitsTrainCompletedEvent
    | DigitsTrainFailedEvent
    | DigitsTrainProgressEvent
)


def encode_event(event: Event) -> str:
    return json.dumps(event, separators=(",", ":"))


class Publisher(Protocol):
    def publish(self, channel: str, message: str) -> int: ...


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if isinstance(v, str) and v.strip() != "" else default


def _publish_event(pub: Publisher | None, channel: str, event: Event) -> None:
    if pub is None:
        return
    try:
        pub.publish(channel, encode_event(event))
    except (OSError, ValueError):
        logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")


def _load_settings() -> Settings:
    """Provider for Settings; tests may monkeypatch this for isolation.

    Keeping this indirection avoids hard ties to TOML/env precedence in tests.
    """
    return Settings.load()


def _build_cfg(payload: DigitsTrainJobV1) -> TrainConfig:
    # Resolve paths via Settings to honor volume mounts and avoid drift.
    s = _load_settings()
    data_root = (s.app.data_root / "mnist").resolve()
    out_dir = (s.app.artifacts_root / "digits" / "models").resolve()
    return TrainConfig(
        data_root=data_root,
        out_dir=out_dir,
        model_id=payload["model_id"],
        epochs=payload["epochs"],
        batch_size=payload["batch_size"],
        lr=float(payload["lr"]),
        weight_decay=1e-2,
        seed=payload["seed"],
        device="cpu",
        optim="adamw",
        scheduler="cosine",
        step_size=10,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=bool(payload["augment"]),
        aug_rotate=10.0,
        aug_translate=0.1,
        noise_prob=0.15,
        dots_prob=0.20,
        dots_count=3,
        dots_size_px=2,
        progress_every_batches=10,
    )


def _run_training(_: TrainConfig) -> Path:
    # Runtime wiring expected; tests monkeypatch this symbol.
    raise RuntimeError("_run_training not wired in this environment")


@dataclass(frozen=True)
class _Context:
    publisher: Publisher | None
    channel: str


def _summarize_training_exception(exc: BaseException) -> str:
    # Memory guard specific message
    if isinstance(exc, RuntimeError) and str(exc) == "memory_pressure_guard_triggered":
        mg = _get_mg_cfg()
        thr = float(mg.threshold_percent)
        return (
            f"Training aborted due to sustained memory pressure (>= {thr:.1f}%). "
            "Reduce batch size or DataLoader workers and retry."
        )
    # Artifact upload error surfaced from worker
    if isinstance(exc, RuntimeError) and "artifact upload failed" in str(exc):
        return "Artifact upload failed: upstream API error. See worker logs for details."
    # Generic: include type and message for clarity
    name = exc.__class__.__name__
    msg = str(exc)
    preview = msg[:300]
    return f"{name}: {preview}" if preview else name


def process_train_job(payload: dict[str, object]) -> None:
    typ = payload.get("type") if isinstance(payload, dict) else None
    if typ != "digits.train.v1":
        _emit_failed(payload, "user", "invalid job type")
        return

    try:
        p: DigitsTrainJobV1 = {
            "type": "digits.train.v1",
            "request_id": str(payload.get("request_id")),
            "user_id": _as_int(payload.get("user_id")),
            "model_id": str(payload.get("model_id")),
            "epochs": _as_int(payload.get("epochs")),
            "batch_size": _as_int(payload.get("batch_size")),
            "lr": _as_float(payload.get("lr")),
            "seed": _as_int(payload.get("seed")),
            "augment": bool(payload.get("augment", False)),
            "notes": (str(payload["notes"]) if isinstance(payload.get("notes"), str) else None),
        }
    except (ValueError, TypeError):
        logging.getLogger("handwriting_ai").info("digits_job_invalid_payload")
        _emit_failed(payload, "user", "invalid payload fields")
        return

    ctx = _make_context()

    try:
        cfg = _build_cfg(p)

        # Publish versioned started event (include resource limits and augmentation for richer UX)
        _ctx = ev.Context(
            request_id=p["request_id"], user_id=p["user_id"], model_id=p["model_id"], run_id=None
        )
        _limits = detect_resource_limits()
        _mem_mb = (
            int(_limits.memory_bytes // (1024 * 1024))
            if isinstance(_limits.memory_bytes, int)
            else None
        )
        _started_v1 = ev.started(
            _ctx,
            total_epochs=p["epochs"],
            cpu_cores=int(_limits.cpu_cores),
            memory_mb=_mem_mb,
            optimal_threads=int(_limits.optimal_threads),
            optimal_workers=int(_limits.optimal_workers),
            max_batch_size=_limits.max_batch_size,
            device=cfg.device,
            batch_size=cfg.batch_size,
            augment=cfg.augment,
            aug_rotate=cfg.aug_rotate,
            aug_translate=cfg.aug_translate,
            noise_prob=float(cfg.noise_prob),
            dots_prob=float(cfg.dots_prob),
        )
        try:
            if ctx.publisher is not None:
                ctx.publisher.publish(ctx.channel, ev.encode_event(_started_v1))
        except (OSError, ValueError):
            logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")
        # Bridge training progress to events via a DI-safe emitter
        _em = _ProgressEmitter(
            publisher=ctx.publisher,
            channel=ctx.channel,
            request_id=p["request_id"],
            user_id=p["user_id"],
            model_id=p["model_id"],
            total_epochs=p["epochs"],
        )
        set_progress_emitter(_em)
        _set_batch_emitter(_em)
        _set_best_emitter(_em)
        _set_epoch_emitter(_em)
        model_dir = _run_training(cfg)
        man = ModelManifest.from_path(model_dir / "manifest.json")
        # Update emitter with run id and publish artifact event
        _em.set_run_id(man.created_at.isoformat())
        try:
            if ctx.publisher is not None:
                _art = ev.artifact(
                    ev.Context(
                        request_id=p["request_id"],
                        user_id=p["user_id"],
                        model_id=p["model_id"],
                        run_id=man.created_at.isoformat(),
                    ),
                    path=str(model_dir),
                )
                ctx.publisher.publish(ctx.channel, ev.encode_event(_art))
        except (OSError, ValueError):
            logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")
        # Versioned completed (no legacy fallback)
        try:
            if ctx.publisher is not None:
                _comp = ev.completed(
                    ev.Context(
                        request_id=p["request_id"],
                        user_id=p["user_id"],
                        model_id=p["model_id"],
                        run_id=man.created_at.isoformat(),
                    ),
                    val_acc=float(man.val_acc),
                )
                ctx.publisher.publish(ctx.channel, ev.encode_event(_comp))
        except (OSError, ValueError):
            logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        _emit_failed(payload, "system", _summarize_training_exception(exc))
        raise
    finally:
        # Ensure emitter cleared to avoid cross-job leakage in long-lived workers
        set_progress_emitter(None)


def _make_context() -> _Context:
    pub = _make_publisher()
    ch = _get_env("DIGITS_EVENTS_CHANNEL", DEFAULT_EVENTS_CHANNEL)
    return _Context(publisher=pub, channel=ch)


def _make_publisher() -> Publisher | None:
    # No-op by default; runtime may override via monkeypatching or custom wiring
    return None


class _ProgressEmitter:
    def __init__(
        self,
        *,
        publisher: Publisher | None,
        channel: str,
        request_id: str,
        user_id: int,
        model_id: str,
        total_epochs: int,
    ) -> None:
        self._publisher = publisher
        self._channel = channel
        self._req = request_id
        self._uid = int(user_id)
        self._mid = model_id
        self._tot = int(total_epochs)
        self._run: str | None = None

    def emit(self, *, epoch: int, total_epochs: int, val_acc: float | None) -> None:
        # No legacy progress event; epoch.v1 is emitted directly from training loop
        _ = (total_epochs, val_acc)

    def set_run_id(self, run_id: str) -> None:
        self._run = run_id

    # Optional emitters wired in training.progress
    def emit_batch(self, metrics: ev.BatchMetrics) -> None:
        """Emit batch progress event.

        Single source of truth: accepts BatchMetrics dataclass.
        """
        if self._publisher is None:
            logging.getLogger("handwriting_ai").warning(
                "batch_event_skipped_no_publisher epoch=%d batch=%d req=%s",
                metrics.epoch,
                metrics.batch,
                self._req,
            )
            return
        try:
            msg = ev.batch(
                ev.Context(
                    request_id=self._req,
                    user_id=self._uid,
                    model_id=self._mid,
                    run_id=self._run,
                ),
                metrics,
            )
            result = self._publisher.publish(self._channel, ev.encode_event(msg))
            logging.getLogger("handwriting_ai").info(
                "batch_event_published epoch=%d batch=%d channel=%s subscribers=%d req=%s",
                metrics.epoch,
                metrics.batch,
                self._channel,
                result,
                self._req,
            )
        except Exception as exc:
            logging.getLogger("handwriting_ai").error(
                "batch_event_publish_failed epoch=%d batch=%d error_type=%s error=%s req=%s",
                metrics.epoch,
                metrics.batch,
                type(exc).__name__,
                str(exc),
                self._req,
                exc_info=True,
            )

    def emit_best(self, *, epoch: int, val_acc: float) -> None:
        try:
            if self._publisher is not None:
                msg = ev.best(
                    ev.Context(
                        request_id=self._req,
                        user_id=self._uid,
                        model_id=self._mid,
                        run_id=self._run,
                    ),
                    epoch=int(epoch),
                    val_acc=float(val_acc),
                )
                self._publisher.publish(self._channel, ev.encode_event(msg))
        except (OSError, ValueError):
            logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")

    def emit_epoch(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_acc: float,
        time_s: float,
    ) -> None:
        try:
            if self._publisher is not None:
                msg = ev.epoch(
                    ev.Context(
                        request_id=self._req,
                        user_id=self._uid,
                        model_id=self._mid,
                        run_id=self._run,
                    ),
                    epoch=int(epoch),
                    total_epochs=int(total_epochs),
                    train_loss=float(train_loss),
                    val_acc=float(val_acc),
                    time_s=float(time_s),
                )
                self._publisher.publish(self._channel, ev.encode_event(msg))
        except (OSError, ValueError):
            logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")


def _as_int(v: object) -> int:
    if isinstance(v, bool):
        raise ValueError("bool not allowed")
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return int(v)
    raise ValueError("invalid int")


def _as_float(v: object) -> float:
    if isinstance(v, bool):
        raise ValueError("bool not allowed")
    if isinstance(v, float):
        return v
    if isinstance(v, int):
        return float(v)
    if isinstance(v, str):
        return float(v)
    raise ValueError("invalid float")


def _emit_failed(
    payload: dict[str, object] | object, kind: Literal["user", "system"], msg: str
) -> None:
    ctx = _make_context()
    req = ""
    uid = 0
    mid = ""
    if isinstance(payload, dict):
        req = str(payload.get("request_id", ""))
        try:
            uid = _as_int(payload.get("user_id"))
        except ValueError:
            logging.getLogger("handwriting_ai").debug("digits_job_user_id_invalid")
            uid = 0
        mid = str(payload.get("model_id", ""))
    # Versioned failed (no legacy fallback)
    try:
        if ctx.publisher is not None:
            _f = ev.failed(
                ev.Context(request_id=req, user_id=uid, model_id=mid, run_id=None),
                error_kind=kind,
                message=msg,
            )
            ctx.publisher.publish(ctx.channel, ev.encode_event(_f))
    except (OSError, ValueError):
        logging.getLogger("handwriting_ai").debug("digits_event_publish_failed")
