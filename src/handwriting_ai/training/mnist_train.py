from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.logging import get_logger, init_logging

from .artifacts import write_artifacts as _write_artifacts_impl
from .calibrate import calibrate_input_pipeline
from .dataset import make_loaders as _make_loaders_impl
from .optim import build_optimizer_and_scheduler as _build_optimizer_and_scheduler
from .progress import ProgressEmitter
from .progress import emit_best as _emit_best
from .progress import emit_epoch as _emit_epoch
from .progress import emit_progress as _emit_progress
from .progress import set_progress_emitter as _set_progress_emitter
from .runtime import apply_threads, build_effective_config
from .train_config import TrainConfig
from .train_types import MNISTLike, TrainableModel
from .train_utils import _apply_affine, _build_model, _configure_threads, _ensure_image, _set_seed

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


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
        t0 = _time.perf_counter()
        log.info(f"epoch_start_{ep}_{cfg.epochs}")
        train_loss = _train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            ep=ep,
            ep_total=cfg.epochs,
            total_batches=len(train_loader),
        )
        val_acc = _evaluate(model, test_loader, device)
        dt = _time.perf_counter() - t0
        log.info(
            f"epoch_done idx={ep} train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} time_s={dt:.1f}"
        )
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
    log = get_logger()
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)
    # Compute and apply effective configuration
    ec, limits = build_effective_config(cfg)
    # Optional empirical preflight calibration
    if cfg.calibrate:
        cache_path = Path("artifacts") / "calibration.json"
        ttl_s = 7 * 24 * 60 * 60
        ec = calibrate_input_pipeline(
            bases[0],
            limits=limits,
            requested_batch_size=int(cfg.batch_size),
            samples=max(1, int(cfg.calibration_samples)),
            cache_path=cache_path,
            ttl_seconds=ttl_s,
            force=bool(cfg.force_calibration),
        )
    apply_threads(ec)
    # Log threading and device configuration
    intra = torch.get_num_threads()
    interop = torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
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
    train_ds, train_loader, test_loader = _make_loaders_impl(train_base, test_base, cfg, loader_cfg)
    # Limit OpenMP/BLAS thread pools alongside ATen threads
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=ec.intra_threads):
        model = _build_model()
        log.info("model_built_starting_training")
        opt, sch = _build_optimizer_and_scheduler(model, cfg)
        try:
            best_sd, best_val = _run_training_loop(
                model, train_loader, test_loader, device, cfg, opt, sch
            )
        except KeyboardInterrupt:
            logging.getLogger("handwriting_ai").info("training_interrupted_by_user")
            best_val = -1.0
            best_sd = None

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


def set_progress_emitter(emitter: ProgressEmitter | None) -> None:
    _set_progress_emitter(emitter)


__all__ = [
    "TrainConfig",
    "MNISTLike",
    "TrainableModel",
    "_apply_affine",
    "_ensure_image",
    "_set_seed",
    "_build_model",
    "_build_optimizer_and_scheduler",
    "_configure_threads",
    "_evaluate",
    "_train_epoch",
    "make_loaders",
    "train_with_config",
    "set_progress_emitter",
]
