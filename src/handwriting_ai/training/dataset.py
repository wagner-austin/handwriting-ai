from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..preprocess import PreprocessOptions, run_preprocess
from .augment import (
    apply_affine,
    ensure_l_mode,
    maybe_add_dots,
    maybe_add_noise,
    maybe_blur,
    maybe_morph,
)


class _TrainCfgProto(Protocol):
    @property
    def augment(self) -> bool: ...

    @property
    def aug_rotate(self) -> float: ...

    @property
    def aug_translate(self) -> float: ...

    @property
    def noise_prob(self) -> float: ...

    @property
    def noise_salt_vs_pepper(self) -> float: ...

    @property
    def dots_prob(self) -> float: ...

    @property
    def dots_count(self) -> int: ...

    @property
    def dots_size_px(self) -> int: ...

    @property
    def blur_sigma(self) -> float: ...

    @property
    def morph(self) -> str: ...

    @property
    def morph_kernel_px(self) -> int: ...

    @property
    def batch_size(self) -> int: ...


class MNISTLike(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


@dataclass(frozen=True)
class _AugmentKnobs:
    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str
    morph_kernel_px: int


def _knobs_from_cfg(cfg: _TrainCfgProto) -> _AugmentKnobs:
    noise_prob = cfg.noise_prob if hasattr(cfg, "noise_prob") else 0.0
    noise_salt_vs_pepper = cfg.noise_salt_vs_pepper if hasattr(cfg, "noise_salt_vs_pepper") else 0.5
    dots_prob = cfg.dots_prob if hasattr(cfg, "dots_prob") else 0.0
    dots_count = cfg.dots_count if hasattr(cfg, "dots_count") else 0
    dots_size_px = cfg.dots_size_px if hasattr(cfg, "dots_size_px") else 1
    blur_sigma = cfg.blur_sigma if hasattr(cfg, "blur_sigma") else 0.0
    morph_mode = cfg.morph if hasattr(cfg, "morph") else "none"
    morph_kernel_px = cfg.morph_kernel_px if hasattr(cfg, "morph_kernel_px") else 1
    return _AugmentKnobs(
        enable=bool(cfg.augment),
        rotate_deg=float(cfg.aug_rotate),
        translate_frac=float(cfg.aug_translate),
        noise_prob=float(noise_prob),
        noise_salt_vs_pepper=float(noise_salt_vs_pepper),
        dots_prob=float(dots_prob),
        dots_count=int(dots_count),
        dots_size_px=int(dots_size_px),
        blur_sigma=float(blur_sigma),
        morph_mode=str(morph_mode),
        morph_kernel_px=int(morph_kernel_px),
    )


def _normalize_morph(x: str) -> Literal["none", "erode", "dilate"]:
    if x == "erode":
        return "erode"
    if x == "dilate":
        return "dilate"
    return "none"


class PreprocessDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset wrapper that applies optional augmentation then service preprocess."""

    def __init__(self, base: MNISTLike, cfg: _TrainCfgProto) -> None:
        self._base = base
        self._opts = PreprocessOptions(
            invert=None, center=True, visualize=False, visualize_max_kb=0
        )
        self._knobs = _knobs_from_cfg(cfg)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_raw, label = self._base[idx]
        g = ensure_l_mode(img_raw)
        if self._knobs.enable:
            g = apply_affine(g, self._knobs.rotate_deg, self._knobs.translate_frac)
            g = maybe_add_noise(g, self._knobs.noise_prob, self._knobs.noise_salt_vs_pepper)
            g = maybe_add_dots(
                g, self._knobs.dots_prob, self._knobs.dots_count, self._knobs.dots_size_px
            )
            g = maybe_blur(g, self._knobs.blur_sigma)
            morph = _normalize_morph(self._knobs.morph_mode)
            g = maybe_morph(g, morph, self._knobs.morph_kernel_px)
        out = run_preprocess(g, self._opts)
        t = out.tensor.squeeze(0)
        return t, torch.tensor(int(label), dtype=torch.long)


def make_loaders(
    train_base: MNISTLike, test_base: MNISTLike, cfg: _TrainCfgProto
) -> tuple[
    Dataset[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    train_ds: Dataset[tuple[Tensor, Tensor]] = PreprocessDataset(train_base, cfg)
    test_ds: Dataset[tuple[Tensor, Tensor]] = PreprocessDataset(test_base, cfg)
    train_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    test_loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    return train_ds, train_loader, test_loader
