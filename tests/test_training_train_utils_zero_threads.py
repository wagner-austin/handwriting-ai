from __future__ import annotations

import torch

from handwriting_ai.training.mnist_train import _configure_threads


class _Cfg:
    @property
    def threads(self) -> int:  # 0 triggers early-return branch
        return 0


def test_configure_threads_zero_noop() -> None:
    before = torch.get_num_threads()
    _configure_threads(_Cfg())
    after = torch.get_num_threads()
    assert after == before
