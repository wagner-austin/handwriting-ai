from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from handwriting_ai.training.augment import ensure_l_mode, maybe_add_dots
from handwriting_ai.training.mnist_train import TrainConfig, make_loaders


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        for y in range(10, 18):
            for x in range(12, 16):
                img.putpixel((x, y), 255)
        return img, idx % 10


def _cfg(tmp: Path) -> TrainConfig:
    return TrainConfig(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-2,
        seed=123,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=True,
        aug_rotate=5.0,
        aug_translate=0.1,
    )


def test_dataset_with_noise_and_blur(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(cfg, noise_prob=0.2, noise_salt_vs_pepper=0.6, blur_sigma=0.5)
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    ds, train_loader, test_loader = make_loaders(train_base, test_base, cfg)
    sample: tuple[Tensor, Tensor] = ds[0]
    x, y = sample
    assert hasattr(x, "shape") and hasattr(y, "shape")
    assert tuple(x.shape[1:]) == (28, 28)


def test_dataset_with_dots_and_morph(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg, dots_prob=1.0, dots_count=2, dots_size_px=2, morph="erode", morph_kernel_px=3
    )
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    ds2, train_loader2, test_loader2 = make_loaders(train_base, test_base, cfg)
    sample2: tuple[Tensor, Tensor] = ds2[0]
    x, y = sample2
    assert hasattr(x, "shape") and hasattr(y, "shape")
    assert tuple(x.shape[1:]) == (28, 28)


def test_ensure_l_mode_converts_rgb() -> None:
    rgb = Image.new("RGB", (8, 8), (0, 0, 0))
    g = ensure_l_mode(rgb)
    assert g.mode == "L"


def test_maybe_add_dots_early_return(monkeypatch: pytest.MonkeyPatch) -> None:
    img = Image.new("L", (10, 10), 0)

    # Force the early return path inside maybe_add_dots (random.random() >= p)
    monkeypatch.setattr("random.random", lambda: 0.99)
    out = maybe_add_dots(img, prob=0.5, count=3, size_px=2)
    # Should be identical shape and remain grayscale
    assert out.size == img.size and out.mode == "L"


def test_dataset_with_morph_dilate(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(cfg, morph="dilate", morph_kernel_px=3)
    train_base = _TinyBase(2)
    test_base = _TinyBase(2)
    ds, train_loader, test_loader = make_loaders(train_base, test_base, cfg)
    x, y = ds[0]
    assert tuple(x.shape[1:]) == (28, 28)
