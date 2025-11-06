from __future__ import annotations

import pytest
import torch

from handwriting_ai.training.mnist_train import _configure_threads


class _Cfg:
    def __init__(self, threads: int) -> None:
        self._threads = threads

    @property
    def threads(self) -> int:
        return self._threads


def test_configure_threads_handles_runtimeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(nthreads: int) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(torch, "set_num_interop_threads", _raise, raising=True)
    _configure_threads(_Cfg(threads=2))
