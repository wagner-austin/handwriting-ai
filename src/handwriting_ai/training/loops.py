from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim.optimizer import Optimizer

from handwriting_ai.logging import get_logger
from handwriting_ai.monitoring import check_memory_pressure, get_memory_stats

from .progress import emit_batch as _emit_batch
from .safety import on_batch_check


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
        # Proactive memory guard: check every batch (not only on log cadence)
        if on_batch_check():
            log.info("mem_guard_abort e=%s b=%s", ep, (batch_idx + 1))  # pragma: no cover
            raise RuntimeError("memory_pressure_guard_triggered")  # pragma: no cover
        total += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
        if batch_idx % log_every == 0:
            avg_loss = loss_sum / total if total > 0 else 0.0
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_acc = float((preds == y).float().mean().item())
            dt = _time.perf_counter() - t0
            ips = (int(y.size(0)) / dt) if dt > 0 else 0.0
            mem = get_memory_stats()
            pressed = check_memory_pressure()
            press = "true" if pressed else "false"
            log.info(
                f"train_batch_done epoch={ep}/{ep_total} "
                f"batch={batch_idx+1}/{total_batches} "
                f"batch_loss={float(loss.item()):.4f} batch_acc={batch_acc:.4f} "
                f"avg_loss={avg_loss:.4f} samples_per_sec={ips:.1f} "
                f"rss_mb={mem.rss_mb} mem_pct={mem.percent:.1f} mem_pressure={press}"
            )
            _emit_batch(
                epoch=ep,
                total_epochs=ep_total,
                batch=(batch_idx + 1),
                total_batches=total_batches,
                batch_loss=float(loss.item()),
                batch_acc=batch_acc,
                avg_loss=avg_loss,
                samples_per_sec=ips,
            )
    return loss_sum / total if total > 0 else 0.0
