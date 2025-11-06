from __future__ import annotations

import types as _types
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.resources import ResourceLimits


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, int(idx % 10)


def _cfg(tmp: Path) -> mt.TrainConfig:
    return mt.TrainConfig(
        data_root=tmp / "data",
        out_dir=tmp / "out",
        model_id="m",
        epochs=1,
        batch_size=8,
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


def test_train_uses_resource_limits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Force resource limits to a known state
    def _fake_detect() -> ResourceLimits:
        return ResourceLimits(
            cpu_cores=4,
            memory_bytes=1024 * 1024 * 1024,
            optimal_threads=2,
            optimal_workers=0,
            max_batch_size=4,
        )

    monkeypatch.setattr("handwriting_ai.training.runtime.detect_resource_limits", _fake_detect)

    # Spy threadpool_limits to ensure it is called with our intra threads
    calls = {"n": 0, "limit": 0}

    class _Ctx(AbstractContextManager[None]):
        def __init__(self, limit: int) -> None:
            self._limit = limit

        def __enter__(self) -> None:
            return None

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: _types.TracebackType | None,
        ) -> Literal[False]:
            return False

    def _fake_threadpool_limits(*, limits: int) -> AbstractContextManager[None]:
        calls["n"] += 1
        calls["limit"] = int(limits)
        return _Ctx(limits)

    # Monkeypatch within module after it imports

    monkeypatch.setattr("threadpoolctl.threadpool_limits", _fake_threadpool_limits)

    cfg = _cfg(tmp_path)
    base = _TinyBase()
    out = mt.train_with_config(cfg, (base, base))
    assert (out / "model.pt").exists()
    # Effective batch capped to 4 per ResourceLimits and persisted in manifest
    man = (out / "manifest.json").read_text(encoding="utf-8")
    assert '"batch_size": 4' in man
    # Torch threads configured to 2
    assert torch.get_num_threads() == 2
    assert calls["n"] >= 1 and calls["limit"] == 2
