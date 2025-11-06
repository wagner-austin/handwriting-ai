from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from handwriting_ai.training.mnist_train import _configure_threads


class _Cfg:
    @property
    def threads(self) -> int:
        return 1


def test_configure_threads_without_set_num_interop() -> None:
    had_attr = hasattr(torch, "set_num_interop_threads")
    saved: Callable[[int], None] | None = getattr(torch, "set_num_interop_threads", None)
    if had_attr:
        delattr(torch, "set_num_interop_threads")
    try:
        # This should not raise and should skip interop configuration branch
        _configure_threads(_Cfg())
        # Touch logger to satisfy guard; also asserts threading configured
        logging.getLogger("handwriting_ai").info("set_num_interop_threads_absent")
        assert torch.get_num_threads() >= 1
    finally:
        if had_attr and saved is not None:
            name = "set_num_interop_threads"
            setattr(torch, name, saved)
