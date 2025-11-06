from __future__ import annotations

import logging
from typing import Protocol

import torch
from PIL import Image

from handwriting_ai.inference.engine import _build_model as _engine_build_model

from .augment import apply_affine as _apply_affine_impl
from .train_types import MNIST_N_CLASSES, TrainableModel


def _apply_affine(img: Image.Image, deg_max: float, tx_frac: float) -> Image.Image:
    return _apply_affine_impl(img, deg_max, tx_frac)


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


class _CfgThreads(Protocol):
    @property
    def threads(self) -> int: ...


def _configure_threads(cfg: _CfgThreads) -> None:
    threads = int(cfg.threads)
    if threads > 0:
        torch.set_num_threads(threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, threads // 2))
            except RuntimeError:
                logging.getLogger("handwriting_ai").info("set_num_interop_threads_failed")
