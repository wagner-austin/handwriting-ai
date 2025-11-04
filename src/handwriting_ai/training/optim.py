from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Protocol

from torch.nn.parameter import Parameter
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.sgd import SGD


class _TrainableModel(Protocol):
    def parameters(self) -> Iterable[Parameter]: ...  # pragma: no cover - typing only


class _Cfg(Protocol):
    @property
    def lr(self) -> float: ...

    @property
    def weight_decay(self) -> float: ...

    @property
    def optim(self) -> str: ...

    @property
    def scheduler(self) -> str: ...

    @property
    def epochs(self) -> int: ...

    @property
    def min_lr(self) -> float: ...

    @property
    def step_size(self) -> int: ...

    @property
    def gamma(self) -> float: ...


if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer


def build_optimizer_and_scheduler(
    model: _TrainableModel, cfg: _Cfg
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
