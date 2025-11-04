from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.mnist_train import (
    TrainConfig,
    _build_model,
    _build_optimizer_and_scheduler,
    _configure_threads,
    _train_epoch,
)


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        # Simple white canvas with a small black square
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        return img, idx % 10


def _base_cfg() -> TrainConfig:
    return TrainConfig(
        data_root=Path("./data/mnist"),
        out_dir=Path("./artifacts/digits/models"),
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-2,
        seed=0,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=10,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_optimizer_scheduler_variants() -> None:
    model = _build_model()
    for opt in ("sgd", "adam", "adamw"):
        for sched in ("none", "cosine", "step"):
            cfg = _base_cfg()
            cfg = replace(cfg, optim=opt, scheduler=sched, epochs=2, step_size=1)
            optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg)
            # Optimizer has param_groups and step/zero_grad
            assert hasattr(optimizer, "param_groups")
            optimizer.zero_grad(set_to_none=True)
            optimizer.step()
            # Scheduler presence by mode
            if sched == "none":
                assert scheduler is None
            else:
                assert scheduler is not None


def test_train_epoch_smoke_with_fake_data() -> None:
    base = _TinyBase(4)
    ds = PreprocessDataset(base, _base_cfg())
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        ds, batch_size=4, shuffle=False
    )
    model = _build_model()
    cfg = _base_cfg()
    optimizer, _ = _build_optimizer_and_scheduler(model, cfg)
    device = torch.device("cpu")
    loss = _train_epoch(
        model,
        loader,
        device,
        optimizer,
        ep=1,
        ep_total=1,
        total_batches=len(loader),
    )
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_augment_flag_yields_valid_sample() -> None:
    base = _TinyBase(1)
    cfg = replace(_base_cfg(), augment=True, aug_rotate=5.0, aug_translate=0.1)
    ds = PreprocessDataset(base, cfg)
    x, y = ds[0]
    assert list(x.shape) == [1, 28, 28]
    assert 0 <= int(y) <= 9


def test_configure_threads_no_crash() -> None:
    cfg = _base_cfg()
    cfg = replace(cfg, threads=1)
    _configure_threads(cfg)
    assert torch.get_num_threads() >= 1
