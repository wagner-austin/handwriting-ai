from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.logging import get_logger, init_logging
from handwriting_ai.monitoring import log_memory_snapshot, log_system_info

from .artifacts import write_artifacts as _write_artifacts_impl
from .calibrate import calibrate_input_pipeline
from .calibration.ds_spec import AugmentSpec, MNISTSpec, PreprocessSpec
from .dataset import make_loaders as _make_loaders_impl
from .optim import build_optimizer_and_scheduler as _build_optimizer_and_scheduler
from .progress import ProgressEmitter
from .progress import emit_best as _emit_best
from .progress import emit_epoch as _emit_epoch
from .progress import emit_progress as _emit_progress
from .progress import set_batch_cadence as _set_batch_cadence
from .progress import set_progress_emitter as _set_progress_emitter
from .resources import ResourceLimits
from .runtime import apply_threads, build_effective_config
from .safety import (
    MemoryGuardConfig,
    compute_memory_guard_config,
    reset_memory_guard,
    set_memory_guard_config,
)
from .train_config import TrainConfig
from .train_types import MNISTLike, TrainableModel
from .train_utils import _apply_affine, _build_model, _configure_threads, _ensure_image, _set_seed

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


def _run_training_loop(
    model: TrainableModel,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    cfg: TrainConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> tuple[dict[str, Tensor] | None, float]:
    log = get_logger()
    best_val = -1.0
    best_sd: dict[str, Tensor] | None = None
    epochs_no_improve = 0
    import time as _time

    for ep in range(1, cfg.epochs + 1):
        # Reset consecutive memory guard at epoch boundary
        reset_memory_guard()
        t0 = _time.perf_counter()
        log_memory_snapshot(context="epoch_start")
        train_loss = _train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            ep=ep,
            ep_total=cfg.epochs,
            total_batches=len(train_loader),
        )
        log_memory_snapshot(context="train_epoch_done")
        val_acc = _evaluate(model, test_loader, device)
        log_memory_snapshot(context="validation_done")
        dt = _time.perf_counter() - t0
        log.info(
            f"epoch_done idx={ep} train_loss={train_loss:.4f} val_acc={val_acc:.4f} time_s={dt:.1f}"
        )
        log_memory_snapshot(context="epoch_done")
        _emit_epoch(
            epoch=ep,
            total_epochs=cfg.epochs,
            train_loss=float(train_loss),
            val_acc=float(val_acc),
            time_s=float(dt),
        )
        cadence = int(cfg.progress_every_epochs) if hasattr(cfg, "progress_every_epochs") else 1
        cadence = max(1, cadence)
        if (ep % cadence == 0) or (ep == cfg.epochs):
            _emit_progress(epoch=ep, total_epochs=cfg.epochs, val_acc=float(val_acc))
        if val_acc > best_val + float(cfg.min_delta):
            best_val = float(val_acc)
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            log.info(f"new_best_val_acc={best_val:.4f}")
            _emit_best(epoch=ep, val_acc=float(best_val))
        else:
            epochs_no_improve += 1
        if scheduler is not None:
            scheduler.step()
        if cfg.patience > 0 and epochs_no_improve >= int(cfg.patience):
            log.info(f"early_stop_at_epoch={ep} no_improve={epochs_no_improve}")
            break
    return best_sd, float(best_val)


def _configure_interop_threads(interp_threads: int | None) -> None:
    # Helper must propagate RuntimeError for tests that validate raising
    if hasattr(torch, "set_num_interop_threads") and interp_threads is not None:
        torch.set_num_interop_threads(int(interp_threads))


def make_loaders(
    train_base: MNISTLike, test_base: MNISTLike, cfg: TrainConfig
) -> tuple[
    Dataset[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    train_ds, train_loader, test_loader = _make_loaders_impl(train_base, test_base, cfg)
    return train_ds, train_loader, test_loader


def _evaluate(
    model: TrainableModel,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> float:
    from .loops import evaluate as _ev

    return _ev(model, loader, device)


def _train_epoch(
    model: TrainableModel,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    optimizer: Optimizer,
    ep: int,
    ep_total: int,
    total_batches: int,
) -> float:
    from .loops import train_epoch as _te

    return _te(
        model,
        train_loader,
        device,
        optimizer,
        ep=ep,
        ep_total=ep_total,
        total_batches=total_batches,
    )


def train_with_config(cfg: TrainConfig, bases: tuple[MNISTLike, MNISTLike]) -> Path:
    # Ensure logger is initialized for scripts/CLI contexts
    init_logging()
    # Surface system info at training start for diagnostics
    log_system_info()
    log = get_logger()
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    # Compute initial effective configuration
    ec, limits = build_effective_config(cfg)

    # Set interop threads once before any parallel work.
    # Calibration varies only intra/loader/batch; interop remains fixed.
    # Apply interop threads; training continues even if setting fails
    def run_forever() -> None:
        try:
            _configure_interop_threads(ec.interop_threads)
        except RuntimeError as exc:
            logging.getLogger("handwriting_ai").error("set_num_interop_threads_failed msg=%s", exc)

    run_forever()
    # Configure memory guard BEFORE calibration to protect against OOM during measurement
    _configure_memory_guard_from_limits(cfg, limits)
    # Always run empirical preflight calibration to avoid heuristic drift.
    # Use out_dir for cache/checkpoint to ensure test isolation
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.out_dir / "calibration.json"
    ttl_s = 7 * 24 * 60 * 60
    # Build explicit dataset spec for calibration (no dataset pickling across processes)
    ds_spec = _build_calibration_spec(cfg)
    ec = calibrate_input_pipeline(
        ds_spec,
        limits=limits,
        requested_batch_size=int(ec.batch_size),
        samples=max(1, int(cfg.calibration_samples)),
        cache_path=cache_path,
        ttl_seconds=ttl_s,
        force=bool(cfg.force_calibration),
    )
    apply_threads(ec)
    # Centralize batch progress frequency control (Discord/consumers rely on this cadence)
    _set_batch_cadence(int(cfg.progress_every_batches))
    try:
        # Log threading and device configuration
        intra = torch.get_num_threads()
        interop = (
            torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
        )
        if interop is not None:
            log.info(f"threads_configured requested={cfg.threads} intra={intra} interop={interop}")
        else:
            log.info(f"threads_configured requested={cfg.threads} intra={intra}")
        log.info(f"device={device}")

        # DataLoader configuration from resource limits
        loader_cfg = ec.loader_cfg
        log.info(
            "dataloader_config "
            f"batch_size={loader_cfg.batch_size} num_workers={loader_cfg.num_workers} "
            f"persistent_workers={loader_cfg.persistent_workers} "
            f"prefetch_factor={loader_cfg.prefetch_factor}"
        )

        train_base, test_base = bases
        # Build loaders with resource-aware configuration
        _train_ds, train_loader, test_loader = _make_loaders_impl(
            train_base, test_base, cfg, loader_cfg
        )
        # Limit OpenMP/BLAS thread pools alongside ATen threads
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=ec.intra_threads):
            model = _build_model()
            log.info("model_built_starting_training")
            opt, sch = _build_optimizer_and_scheduler(model, cfg)
            best_sd: dict[str, Tensor] | None = None
            best_val: float = -1.0
            try:
                best_sd, best_val = _run_training_loop(
                    model, train_loader, test_loader, device, cfg, opt, sch
                )
            except KeyboardInterrupt as exc:
                logging.getLogger("handwriting_ai").error(
                    "training_interrupted_by_user error=%s", exc
                )
                # Gracefully handle interrupt: save current progress instead of losing work
                best_sd = model.state_dict()
                best_val = _evaluate(model, test_loader, device)

        log.info("threadpoolctl_applied")

        # Write artifacts via helper
        sd = best_sd if best_sd is not None else model.state_dict()
        val = float(best_val if best_val >= 0 else _evaluate(model, test_loader, device))
        model_dir = _write_artifacts_impl(
            out_dir=cfg.out_dir,
            model_id=cfg.model_id,
            model_state=sd,
            epochs=cfg.epochs,
            batch_size=loader_cfg.batch_size,
            lr=cfg.lr,
            seed=cfg.seed,
            device_str=cfg.device,
            optim=cfg.optim,
            scheduler=cfg.scheduler,
            augment=cfg.augment,
            test_val_acc=val,
        )
        log.info(f"artifact_written_to={model_dir}")
        return model_dir
    finally:
        # Avoid cross-run/test leakage of cadence
        _set_batch_cadence(0)


def set_progress_emitter(emitter: ProgressEmitter | None) -> None:
    _set_progress_emitter(emitter)


def _build_calibration_spec(cfg: TrainConfig) -> PreprocessSpec:
    aug_spec = AugmentSpec(
        augment=bool(cfg.augment),
        aug_rotate=float(cfg.aug_rotate),
        aug_translate=float(cfg.aug_translate),
        noise_prob=float(cfg.noise_prob),
        noise_salt_vs_pepper=float(cfg.noise_salt_vs_pepper),
        dots_prob=float(cfg.dots_prob),
        dots_count=int(cfg.dots_count),
        dots_size_px=int(cfg.dots_size_px),
        blur_sigma=float(cfg.blur_sigma),
        morph=str(cfg.morph),
    )
    mnist_spec = MNISTSpec(root=cfg.data_root, train=True)
    return PreprocessSpec(base_kind="mnist", mnist=mnist_spec, inline=None, augment=aug_spec)


def _memory_tier_name(memory_bytes: int | None) -> str:
    """Return human-readable tier name for logging."""
    if memory_bytes is None:
        return "unlimited"
    gb = float(memory_bytes) / (1024.0 * 1024.0 * 1024.0)
    if gb < 1.0:
        return "sub_1gb"
    if gb < 2.0:
        return "1_2gb"
    if gb < 4.0:
        return "2_4gb"
    return "4gb_plus"


def _configure_memory_guard_from_limits(cfg: TrainConfig, limits: ResourceLimits) -> None:
    """Configure memory guard from available memory. Threshold is computed, not configurable."""
    if not cfg.memory_guard:
        # User explicitly disabled
        set_memory_guard_config(
            MemoryGuardConfig(enabled=False, threshold_percent=0.0, required_consecutive=0)
        )
        log = get_logger()
        log.info("mem_guard_config enabled=False user_disabled=True")
        return

    # Compute from available memory
    config = compute_memory_guard_config(limits.memory_bytes)
    set_memory_guard_config(config)

    log = get_logger()
    mem_gb = (limits.memory_bytes / (1024.0**3)) if limits.memory_bytes else 0.0
    log.info(
        "mem_guard_config enabled=True threshold_percent=%.1f required_consecutive=%d "
        "memory_gb=%.2f tier=%s",
        config.threshold_percent,
        config.required_consecutive,
        mem_gb,
        _memory_tier_name(limits.memory_bytes),
    )


__all__ = [
    "MNISTLike",
    "TrainConfig",
    "TrainableModel",
    "_apply_affine",
    "_build_model",
    "_build_optimizer_and_scheduler",
    "_configure_interop_threads",
    "_configure_memory_guard_from_limits",
    "_configure_threads",
    "_ensure_image",
    "_evaluate",
    "_set_seed",
    "_train_epoch",
    "make_loaders",
    "set_progress_emitter",
    "train_with_config",
]
