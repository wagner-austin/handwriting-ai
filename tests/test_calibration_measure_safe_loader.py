from __future__ import annotations

from PIL import Image
from torch.utils.data import Dataset

import handwriting_ai.training.calibrate as cal
from handwriting_ai.training.dataset import DataLoaderConfig, PreprocessDataset


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), int(idx)


class _Cfg:
    augment = False
    aug_rotate = 0.0
    aug_translate = 0.0
    noise_prob = 0.0
    noise_salt_vs_pepper = 0.5
    dots_prob = 0.0
    dots_count = 0
    dots_size_px = 1
    blur_sigma = 0.0
    morph = "none"
    morph_kernel_px = 1
    batch_size = 1


def test_safe_loader_workers_positive_branch() -> None:
    ds: PreprocessDataset = PreprocessDataset(_TinyBase(), _Cfg())
    cfg = DataLoaderConfig(
        batch_size=2,
        num_workers=1,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    loader = cal._safe_loader(ds, cfg)
    # Exercising the path where num_workers > 0 (prefetch/persistent set)
    assert loader.batch_size == 2
    assert loader.num_workers == 1
