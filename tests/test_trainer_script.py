from __future__ import annotations

from typing import Protocol

from PIL import Image
from scripts.train_mnist_resnet18 import MNISTLike, _PreprocessDataset


class _Base(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[Image.Image, int]: ...


class _FakeMNIST:
    def __init__(self, n: int = 3) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (32, 32), 255)
        for y in range(12, 20):
            for x in range(14, 18):
                img.putpixel((x, y), 0)
        label = idx % 10
        return img, label


def test_preprocess_dataset_shapes() -> None:
    base: MNISTLike = _FakeMNIST(2)
    ds = _PreprocessDataset(base)
    x0, y0 = ds[0]
    assert list(x0.shape) == [1, 28, 28]
    assert isinstance(int(y0), int)
