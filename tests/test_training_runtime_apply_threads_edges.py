from __future__ import annotations

import torch

from handwriting_ai.training.dataset import DataLoaderConfig
from handwriting_ai.training.runtime import EffectiveConfig, apply_threads


def test_apply_threads_with_no_interop() -> None:
    ec = EffectiveConfig(
        intra_threads=1,
        interop_threads=None,
        batch_size=1,
        loader_cfg=DataLoaderConfig(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
        ),
    )
    apply_threads(ec)
    assert torch.get_num_threads() >= 1
