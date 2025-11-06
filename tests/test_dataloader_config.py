from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Dataset

from handwriting_ai.training.dataset import DataLoaderConfig, make_loaders
from handwriting_ai.training.mnist_train import TrainConfig


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), int(idx)


def _cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        data_root=tmp_path / "data",
        out_dir=tmp_path / "out",
        model_id="m",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-2,
        seed=0,
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


def test_dataloader_config_validation() -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=0,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=-1,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2,
        )
    with pytest.raises(ValueError):
        DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=-1,
        )


def test_make_loaders_with_loader_cfg(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase()
    test_base = _TinyBase()
    lc = DataLoaderConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    _, train_loader, test_loader = make_loaders(train_base, test_base, cfg, lc)
    assert train_loader.batch_size == 2
    assert test_loader.batch_size == 2
    assert train_loader.num_workers == 0 and test_loader.num_workers == 0


def test_make_loaders_with_workers_positive(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase()
    test_base = _TinyBase()
    lc = DataLoaderConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    _, train_loader, test_loader = make_loaders(train_base, test_base, cfg, lc)
    assert train_loader.batch_size == 2 and test_loader.batch_size == 2
    assert train_loader.num_workers == 1 and test_loader.num_workers == 1
