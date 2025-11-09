from __future__ import annotations

from collections.abc import Callable
from typing import overload

import pytest
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

import handwriting_ai.training.loops as loops
from handwriting_ai.events.digits import BatchMetrics


class _Model:
    def train(self) -> object:
        return self

    def eval(self) -> object:
        return self

    def __call__(self, x: Tensor) -> Tensor:
        b = int(x.shape[0])
        # Produce a tensor that participates in autograd to allow backward()
        return torch.zeros((b, 10), dtype=x.dtype, requires_grad=True)


class _NoopOpt(Optimizer):
    def __init__(self) -> None:
        p = torch.nn.Parameter(torch.zeros(1))
        defaults: dict[str, float] = {}
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


def _loader(n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(n):
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(low=0, high=10, size=(8,), dtype=torch.long)
        out.append((x, y))
    return out


def test_train_epoch_emits_last_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Capture batches emitted by loops
    seen: list[int] = []

    def _cap_emit_batch(metrics: BatchMetrics) -> None:
        seen.append(metrics.batch)

    monkeypatch.setattr(loops, "_emit_batch", _cap_emit_batch, raising=True)
    monkeypatch.setattr(loops, "on_batch_check", lambda: False, raising=True)

    model = _Model()
    opt = _NoopOpt()
    device = torch.device("cpu")
    total = 12  # not a multiple of 10 to exercise last-batch emission
    data = _loader(total)
    loss = loops.train_epoch(
        model=model,
        train_loader=data,
        device=device,
        optimizer=opt,
        ep=1,
        ep_total=1,
        total_batches=len(data),
    )
    assert loss >= 0.0
    assert seen[-1] == total
