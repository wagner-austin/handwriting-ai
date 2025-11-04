from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.inference.engine import (
    TorchModel,
)
from handwriting_ai.inference.engine import (
    _build_model as _engine_build_model,
)
from handwriting_ai.logging import get_logger, init_logging

from .artifacts import write_artifacts as _write_artifacts_impl
from .augment import apply_affine as _apply_affine_impl
from .dataset import make_loaders as _make_loaders_impl
from .loops import evaluate as _evaluate_impl
from .loops import train_epoch as _train_epoch_impl
from .optim import build_optimizer_and_scheduler as _build_optimizer_and_scheduler_impl
from .progress import (
    ProgressEmitter,
)
from .progress import (
    emit_progress as _emit_progress,
)
from .progress import (
    set_progress_emitter as _set_progress_emitter,
)


class TrainableModel(TorchModel, Protocol):
    def train(self) -> object: ...
    def state_dict(self) -> dict[str, Tensor]: ...
    def parameters(self) -> Iterable[Parameter]: ...


MNIST_N_CLASSES: Final[int] = 10


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    out_dir: Path
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    optim: str
    scheduler: str
    step_size: int
    gamma: float
    min_lr: float
    patience: int
    min_delta: float
    threads: int
    augment: bool
    aug_rotate: float
    aug_translate: float
    # Optional augmentation modifiers (default-off)
    noise_prob: float = 0.0
    noise_salt_vs_pepper: float = 0.5
    dots_prob: float = 0.0
    dots_count: int = 0
    dots_size_px: int = 1
    blur_sigma: float = 0.0
    morph: str = "none"
    morph_kernel_px: int = 1


class MNISTLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


def _apply_affine(img: Image.Image, deg_max: float, tx_frac: float) -> Image.Image:
    return _apply_affine_impl(img, deg_max, tx_frac)


def _ensure_image(obj: object) -> Image.Image:
    if not isinstance(obj, Image.Image):
        raise RuntimeError("MNIST returned a non-image sample")
    return obj


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def _build_model() -> TrainableModel:
    return _engine_build_model("resnet18", MNIST_N_CLASSES)


def _evaluate(
    model: TrainableModel,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> float:
    return _evaluate_impl(model, loader, device)


def _configure_threads(cfg: TrainConfig) -> None:
    if cfg.threads and cfg.threads > 0:
        torch.set_num_threads(int(cfg.threads))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, int(cfg.threads) // 2))


if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


def _build_optimizer_and_scheduler(
    model: TrainableModel, cfg: TrainConfig
) -> tuple[Optimizer, LRScheduler | None]:
    return _build_optimizer_and_scheduler_impl(model, cfg)


def make_loaders(
    train_base: MNISTLike, test_base: MNISTLike, cfg: TrainConfig
) -> tuple[
    Dataset[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    train_ds, train_loader, test_loader = _make_loaders_impl(train_base, test_base, cfg)
    return train_ds, train_loader, test_loader


def _train_epoch(
    model: TrainableModel,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    optimizer: Optimizer,
    ep: int,
    ep_total: int,
    total_batches: int,
) -> float:
    return _train_epoch_impl(
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
    _configure_threads(cfg)
    # Log threading configuration
    import os

    log.info(f"cpu_cores={os.cpu_count()}")
    intra = torch.get_num_threads()
    interop = torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else None
    if interop is not None:
        log.info(f"threads_configured requested={cfg.threads} intra={intra} interop={interop}")
    else:
        log.info(f"threads_configured requested={cfg.threads} intra={intra}")
    log.info(f"device={device}")

    train_base, test_base = bases
    train_ds, train_loader, test_loader = make_loaders(train_base, test_base, cfg)
    model = _build_model()
    log.info("model_built_starting_training")
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)

    best_val = -1.0
    best_sd: dict[str, Tensor] | None = None
    epochs_no_improve = 0
    try:
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
            _emit_progress(epoch=ep, total_epochs=cfg.epochs, val_acc=float(val_acc))
            if val_acc > best_val + float(cfg.min_delta):
                best_val = float(val_acc)
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                log.info(f"new_best_val_acc={best_val:.4f}")
            else:
                epochs_no_improve += 1
            if scheduler is not None:
                scheduler.step()
            if cfg.patience > 0 and epochs_no_improve >= int(cfg.patience):
                log.info(f"early_stop_at_epoch={ep} no_improve={epochs_no_improve}")
                break
    except KeyboardInterrupt:
        logging.getLogger("handwriting_ai").info("training_interrupted_by_user")

    # Write artifacts via helper
    sd = best_sd if best_sd is not None else model.state_dict()
    val = float(best_val if best_val >= 0 else _evaluate(model, test_loader, device))
    model_dir = _write_artifacts_impl(
        out_dir=cfg.out_dir,
        model_id=cfg.model_id,
        model_state=sd,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
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
