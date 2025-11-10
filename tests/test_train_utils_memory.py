from __future__ import annotations

from collections.abc import Iterable

import pytest

from handwriting_ai.training.train_utils import (
    bytes_of_model_and_grads,
    torch_allocator_stats,
)


class _FakeTensor:
    def __init__(self, num: int, size: int, grad: _FakeTensor | None = None) -> None:
        self._n = int(num)
        self._s = int(size)
        self.grad = grad

    def numel(self) -> int:
        return self._n

    def element_size(self) -> int:
        return self._s


class _FakeModel:
    def __init__(self, params: list[_FakeTensor]) -> None:
        self._params = params

    def parameters(self) -> Iterable[_FakeTensor]:
        return list(self._params)


def test_bytes_of_model_and_grads() -> None:
    # Two params: (100 elems x 4B), (50 elems x 8B) with grad only on second
    p1 = _FakeTensor(100, 4, grad=None)
    g2 = _FakeTensor(50, 8, grad=None)
    p2 = _FakeTensor(50, 8, grad=g2)
    m = _FakeModel([p1, p2])

    param_b, grad_b = bytes_of_model_and_grads(m)
    assert param_b == (100 * 4) + (50 * 8)
    assert grad_b == (50 * 8)


def test_torch_allocator_stats_sane() -> None:
    available, allocated, reserved, max_alloc = torch_allocator_stats()
    if not available:
        assert allocated == 0 and reserved == 0 and max_alloc == 0
    else:
        assert allocated >= 0 and reserved >= 0 and max_alloc >= 0


def test_torch_allocator_stats_cuda_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate CUDA available and provide deterministic values
    import torch

    def _is_available() -> bool:
        return True

    def _current_device() -> int:
        return 0

    def _mem_alloc(dev: int) -> int:
        return 123456

    def _mem_reserved(dev: int) -> int:
        return 234567

    def _mem_max_alloc(dev: int) -> int:
        return 345678

    monkeypatch.setattr(torch.cuda, "is_available", _is_available, raising=True)
    monkeypatch.setattr(torch.cuda, "current_device", _current_device, raising=True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", _mem_alloc, raising=True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", _mem_reserved, raising=True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _mem_max_alloc, raising=True)

    available, allocated, reserved, max_alloc = torch_allocator_stats()
    assert available is True
    assert allocated == 123456
    assert reserved == 234567
    assert max_alloc == 345678
