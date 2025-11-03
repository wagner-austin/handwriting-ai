from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.training.mnist_train import (
    TrainConfig,
    _apply_affine,
    _build_model,
    _build_optimizer_and_scheduler,
    _ensure_image,
    _evaluate,
    _PreprocessDataset,
    _set_seed,
    _train_epoch,
    make_loaders,
    train_with_config,
)


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        return img, idx % 10


def _cfg(tmp: Path) -> TrainConfig:
    return TrainConfig(
        data_root=tmp / "data",  # not used when injecting bases
        out_dir=tmp / "out",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=4,
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
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_apply_affine_identity() -> None:
    img = Image.new("L", (32, 32), 255)
    img.putpixel((16, 16), 0)
    out = _apply_affine(img, deg_max=0.0, tx_frac=0.0)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    # center pixel remains dark
    assert out.getpixel((16, 16)) == 0


def test_set_seed_reproducible() -> None:
    _set_seed(123)
    a = torch.rand(3)
    _set_seed(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_build_model_and_evaluate_smoke() -> None:
    model = _build_model()
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    y = model(x)
    assert tuple(y.shape) == (1, 10)
    # Evaluate on tiny loader
    base = _TinyBase(2)
    ds = _PreprocessDataset(base)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(ds, batch_size=2, shuffle=False)
    acc = _evaluate(model, loader, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_make_loaders_and_train_epoch_augment(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(cfg, augment=True, aug_rotate=5.0, aug_translate=0.1)
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    ds, train_loader, _ = make_loaders(train_base, test_base, cfg)
    assert isinstance(ds, _PreprocessDataset)
    # training DataLoader yields Tensor labels
    model = _build_model()
    opt, _ = _build_optimizer_and_scheduler(model, cfg)
    loss = _train_epoch(model, train_loader, torch.device("cpu"), opt)
    assert isinstance(loss, float)


def test_train_with_config_writes_artifacts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(cfg, out_dir=tmp_path / "out")
    train_base = _TinyBase(4)
    test_base = _TinyBase(2)
    out_dir = train_with_config(cfg, (train_base, test_base))
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "manifest.json").exists()


def test_train_with_scheduler_and_early_stop(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg = replace(
        cfg,
        out_dir=tmp_path / "out_step",
        epochs=2,
        scheduler="step",
        step_size=1,
        patience=1,
        min_delta=0.01,
    )
    train_base = _TinyBase(6)
    test_base = _TinyBase(2)
    out_dir = train_with_config(cfg, (train_base, test_base))
    assert (out_dir / "model.pt").exists()


def test_ensure_image_guard_and_ok() -> None:
    ok_img = Image.new("L", (8, 8), 0)
    img = _ensure_image(ok_img)
    assert isinstance(img, Image.Image)
    raised = False
    try:
        _ensure_image(42)
    except RuntimeError:
        raised = True
    assert raised is True
