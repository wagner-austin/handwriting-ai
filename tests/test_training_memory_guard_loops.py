from __future__ import annotations

from collections.abc import Callable
from typing import overload

import pytest
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from handwriting_ai.training import loops


class _Model:
    def train(self) -> object:
        return self

    def eval(self) -> object:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        b = int(x.shape[0])
        return torch.zeros((b, 10), dtype=x.dtype)


def _loader(n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n):
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(low=0, high=10, size=(8,), dtype=torch.long)
        out.append((x, y))
    return out


class _NoopOpt(Optimizer):
    def __init__(self) -> None:
        defaults: dict[str, float] = {}
        # Provide a dummy parameter to satisfy Optimizer base class
        p = torch.nn.Parameter(torch.zeros(1))
        super().__init__([p], defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...  # pragma: no cover - typing only

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...  # pragma: no cover - typing only

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is not None:
            return float(closure())
        return None

    def zero_grad(self, set_to_none: bool = False) -> None:
        return None


def test_train_epoch_triggers_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the guard path in loops
    monkeypatch.setattr(loops, "on_batch_check", lambda: True, raising=True)

    model = _Model()
    opt = _NoopOpt()
    device = torch.device("cpu")
    data = _loader(10)  # 10batches -> guard check at first batch
    with pytest.raises(RuntimeError):
        loops.train_epoch(
            model=model,
            train_loader=data,
            device=device,
            optimizer=opt,
            ep=1,
            ep_total=1,
            total_batches=len(data),
        )
