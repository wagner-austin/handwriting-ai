from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.inference.engine import (
    TorchModel,
)
from handwriting_ai.inference.engine import (
    _build_model as _engine_build_model,
)
from handwriting_ai.logging import get_logger, init_logging
from handwriting_ai.preprocess import PreprocessOptions, preprocess_signature, run_preprocess


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


class MNISTLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


class _PreprocessDataset(Dataset[tuple[Tensor, Tensor]]):
    """MNIST dataset applying the service preprocess to prevent drift."""

    def __init__(
        self,
        base: MNISTLike,
        augment: bool = False,
        aug_rotate: float = 0.0,
        aug_translate: float = 0.0,
    ) -> None:
        self._base = base
        self._opts = PreprocessOptions(
            invert=None,
            center=True,
            visualize=False,
            visualize_max_kb=0,
        )
        self._augment = bool(augment)
        self._aug_rotate = float(aug_rotate)
        self._aug_translate = float(aug_translate)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_raw, label = self._base[idx]
        img = _ensure_image(img_raw)
        img2 = _apply_affine(img, self._aug_rotate, self._aug_translate) if self._augment else img
        out = run_preprocess(img2, self._opts)
        t = out.tensor.squeeze(0)
        return t, torch.tensor(int(label), dtype=torch.long)


def _apply_affine(img: Image.Image, deg_max: float, tx_frac: float) -> Image.Image:
    import random

    d = max(0.0, float(deg_max))
    t = max(0.0, min(0.5, float(tx_frac)))
    angle = random.uniform(-d, d)
    dx = int(round(t * img.width))
    dy = int(round(t * img.height))
    tx = random.randint(-dx, dx)
    ty = random.randint(-dy, dy)
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    translated = Image.new("L", rotated.size, 0)
    translated.paste(rotated, (tx, ty))
    return translated


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
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            preds = logits.argmax(dim=1)
            correct += int((preds.cpu() == y).sum().item())
            total += y.size(0)
    return (correct / total) if total > 0 else 0.0


def _configure_threads(cfg: TrainConfig) -> None:
    if cfg.threads and cfg.threads > 0:
        torch.set_num_threads(int(cfg.threads))
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(max(1, int(cfg.threads) // 2))


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


def _build_optimizer_and_scheduler(
    model: TrainableModel, cfg: TrainConfig
) -> tuple[Optimizer, LRScheduler | None]:
    if cfg.optim == "sgd":
        optimizer: Optimizer = SGD(
            model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    elif cfg.optim == "adam":
        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        scheduler: LRScheduler | None
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs), eta_min=cfg.min_lr)
    elif cfg.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, cfg.step_size), gamma=cfg.gamma)
    else:
        scheduler = None
    return optimizer, scheduler


def make_loaders(
    train_base: MNISTLike, test_base: MNISTLike, cfg: TrainConfig
) -> tuple[
    Dataset[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    train_ds: Dataset[tuple[Tensor, Tensor]] = _PreprocessDataset(
        train_base, augment=cfg.augment, aug_rotate=cfg.aug_rotate, aug_translate=cfg.aug_translate
    )
    test_ds: Dataset[tuple[Tensor, Tensor]] = _PreprocessDataset(test_base)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
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
    import time as _time

    log = get_logger()
    model.train()
    total = 0
    loss_sum = 0.0
    log_every = 10
    for batch_idx, (x, y) in enumerate(train_loader):
        t0 = _time.perf_counter()
        if batch_idx % log_every == 0:
            log.debug(f"train_loading_batch idx={batch_idx}")
        x = x.to(device)
        y = y.to(device)
        if batch_idx % log_every == 0:
            log.debug("train_forward")
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if batch_idx % log_every == 0:
            log.debug("train_backward")
        torch.autograd.backward((loss,))
        if batch_idx % log_every == 0:
            log.debug("train_step")
        optimizer.step()
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        if batch_idx % log_every == 0:
            avg_loss = loss_sum / total if total > 0 else 0.0
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_acc = float((preds == y).float().mean().item())
            dt = _time.perf_counter() - t0
            ips = (int(y.size(0)) / dt) if dt > 0 else 0.0
            log.info(
                f"train_batch_done epoch={ep}/{ep_total} "
                f"batch={batch_idx+1}/{total_batches} "
                f"batch_loss={float(loss.item()):.4f} batch_acc={batch_acc:.4f} "
                f"avg_loss={avg_loss:.4f} samples_per_sec={ips:.1f}"
            )
    return loss_sum / total if total > 0 else 0.0


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

    # Write artifact
    model_dir = cfg.out_dir / cfg.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    sd = best_sd if best_sd is not None else model.state_dict()
    torch.save(sd, (model_dir / "model.pt").as_posix())
    manifest = {
        "schema_version": "v1",
        "model_id": cfg.model_id,
        "arch": "resnet18",
        "n_classes": MNIST_N_CLASSES,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": float(best_val if best_val >= 0 else _evaluate(model, test_loader, device)),
        "temperature": 1.0,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    log.info(f"artifact_written_to={model_dir}")
    return model_dir
