from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..dataset import _TrainCfgProto as _AugCfgProto

BaseKind = Literal["mnist", "inline"]


@dataclass(frozen=True)
class MNISTSpec:
    root: Path
    train: bool


@dataclass(frozen=True)
class InlineSpec:
    # Lightweight, test-only dataset description
    n: int
    sleep_s: float
    fail: bool


@dataclass(frozen=True)
class AugmentSpec:
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph: str


@dataclass(frozen=True)
class PreprocessSpec:
    base_kind: BaseKind
    mnist: MNISTSpec | None
    inline: InlineSpec | None
    augment: AugmentSpec


def _unused_make_augment_spec(_: _AugCfgProto) -> AugmentSpec:
    # Kept only for potential future use; avoid dynamic getattr that confuses mypy strict.
    # Prefer constructing AugmentSpec directly from known typed structures (e.g., dataset knobs).
    raise NotImplementedError


__all__ = [
    "BaseKind",
    "MNISTSpec",
    "InlineSpec",
    "AugmentSpec",
    "PreprocessSpec",
    "_unused_make_augment_spec",
]
