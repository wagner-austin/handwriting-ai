from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, overload, runtime_checkable

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
        # Interop threads are set once early by the training runtime; avoid re-setting here


class _TensorLike(Protocol):
    def numel(self) -> int: ...  # pragma: no cover - typing only
    def element_size(self) -> int: ...  # pragma: no cover - typing only

    grad: _TensorLike | None  # pragma: no cover - typing only


@runtime_checkable
class _ParamProvider(Protocol):
    def parameters(self) -> Iterable[_TensorLike]: ...  # pragma: no cover - typing only


@overload
def bytes_of_model_and_grads(model: _ParamProvider) -> tuple[int, int]: ...


@overload
def bytes_of_model_and_grads(model: object) -> tuple[int, int]: ...


def bytes_of_model_and_grads(model: object) -> tuple[int, int]:
    """Compute raw bytes for model parameters and their gradients.

    Returns a tuple: (parameter_bytes, gradient_bytes).
    """
    param_bytes = 0
    grad_bytes = 0
    if not isinstance(model, _ParamProvider):
        return 0, 0
    for p in model.parameters():
        pn = int(p.numel()) * int(p.element_size())
        param_bytes += pn
        g = p.grad
        if g is not None:
            gn = int(g.numel()) * int(g.element_size())
            grad_bytes += gn
    return param_bytes, grad_bytes


def torch_allocator_stats() -> tuple[bool, int, int, int]:
    """Return CUDA allocator stats if available: (available, allocated, reserved, max_allocated).

    Values are bytes; when CUDA is unavailable or on CPU-only builds, returns
    (False, 0, 0, 0). Safe across environments.
    """
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return False, 0, 0, 0
    dev = torch.cuda.current_device()
    allocated = int(torch.cuda.memory_allocated(dev))
    reserved = int(torch.cuda.memory_reserved(dev))
    max_alloc = int(torch.cuda.max_memory_allocated(dev))
    return True, allocated, reserved, max_alloc
