from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim.optimizer import Optimizer

from handwriting_ai.logging import get_logger


class _TrainableModel(Protocol):
    def train(self) -> object: ...  # pragma: no cover - typing only
    def eval(self) -> object: ...  # pragma: no cover - typing only
    def __call__(self, x: Tensor) -> Tensor: ...  # pragma: no cover - typing only


def evaluate(
    model: _TrainableModel, loader: Iterable[tuple[Tensor, Tensor]], device: torch.device
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


def train_epoch(
    model: _TrainableModel,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
    optimizer: Optimizer,
    *,
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
