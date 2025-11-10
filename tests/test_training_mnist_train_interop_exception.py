from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.mnist_train import TrainConfig, train_with_config


@pytest.fixture(autouse=True)
def _mock_monitoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock monitoring functions that fail on non-container systems."""
    import handwriting_ai.training.mnist_train as mt

    monkeypatch.setattr(mt, "log_system_info", lambda: None, raising=False)


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __init__(self, n: int = 4) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
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


def test_train_with_interop_threads_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    train_base = _TinyBase(2)
    test_base = _TinyBase(1)

    # Speed up training loop to avoid heavy compute
    def _ok_train_epoch(
        model: mt.TrainableModel,
        train_loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
        optimizer: object,
        ep: int,
        ep_total: int,
        total_batches: int,
    ) -> float:
        return 0.0

    monkeypatch.setattr(mt, "_train_epoch", _ok_train_epoch, raising=True)

    # Force branch: interop_threads is not None and set_num_interop_threads raises
    def _raise_runtime_error(n: int) -> None:
        _ = n
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "set_num_interop_threads", _raise_runtime_error, raising=True)

    out = train_with_config(cfg, (train_base, test_base))
    assert (out / "model.pt").exists()
    assert (out / "manifest.json").exists()
