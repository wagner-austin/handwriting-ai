from __future__ import annotations

import gzip
from pathlib import Path
from typing import Protocol

import pytest
from PIL import Image

from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, MNISTSpec, PreprocessSpec
from handwriting_ai.training.calibration.runner import (
    _build_dataset_from_spec,
    _mnist_read_images_labels,
    _to_spec,
)
from handwriting_ai.training.dataset import PreprocessDataset


class _TinyBase:
    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), color=0), int(idx % 10)


class _CfgProto(Protocol):
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
    morph_kernel_px: int
    batch_size: int


class _Cfg:
    def __init__(self) -> None:
        self.augment = True
        self.aug_rotate = 5.0
        self.aug_translate = 0.1
        self.noise_prob = 0.2
        self.noise_salt_vs_pepper = 0.6
        self.dots_prob = 0.1
        self.dots_count = 2
        self.dots_size_px = 1
        self.blur_sigma = 0.5
        self.morph = "none"
        self.morph_kernel_px = 1
        self.batch_size = 1


def _write_gzip(path: Path, data: bytes) -> None:
    with gzip.open(path, "wb") as f:
        f.write(data)


def _mk_images_header(n: int, rows: int = 28, cols: int = 28, magic: int = 2051) -> bytes:
    return (
        magic.to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + int(rows).to_bytes(4, "big")
        + int(cols).to_bytes(4, "big")
    )


def _mk_labels_header(n: int, magic: int = 2049) -> bytes:
    return magic.to_bytes(4, "big") + int(n).to_bytes(4, "big")


def test_to_spec_from_dataset() -> None:
    base = _TinyBase(5)
    ds = PreprocessDataset(base, _Cfg())
    spec = _to_spec(ds)
    assert isinstance(spec, PreprocessSpec)
    assert spec.base_kind == "inline"
    assert spec.inline is not None and spec.inline.n == len(ds)
    # Ensure knobs are mapped
    assert spec.augment.augment is True and spec.augment.aug_rotate == pytest.approx(5.0)


def test_to_spec_pass_through_for_spec() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=3, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    # Identity for spec input
    assert _to_spec(spec) is spec


def test_build_dataset_from_spec_inline_success() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=7, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds = _build_dataset_from_spec(spec)
    assert isinstance(ds, PreprocessDataset)
    assert len(ds) == 7


def test_build_dataset_from_spec_inline_missing_details() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=None,
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    with pytest.raises(RuntimeError, match="inline spec missing details"):
        _build_dataset_from_spec(spec)


def test_mnist_read_images_labels_happy_path(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 3
    img_header = _mk_images_header(n)
    lbl_header = _mk_labels_header(n)
    images = bytes([0] * (n * 28 * 28))
    labels = bytes([i % 10 for i in range(n)])
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + images)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", lbl_header + labels)
    imgs, lbls = _mnist_read_images_labels(tmp_path, True)
    assert len(imgs) == n and len(lbls) == n


def test_mnist_read_images_labels_missing_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MNIST raw files not found"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_images_header(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", b"bad")  # < 16 bytes
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST images header"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_images_magic(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    img_header = _mk_images_header(1, rows=28, cols=28, magic=9999)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + bytes([0] * 784))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST images file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_truncated_images(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 2
    img_header = _mk_images_header(n)
    # Missing some bytes
    images = bytes([0] * (n * 28 * 28 - 1))
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + images)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n) + bytes([0, 1]))
    with pytest.raises(RuntimeError, match="truncated MNIST images file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_labels_header(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 1
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * 784))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", b"bad")
    with pytest.raises(RuntimeError, match="invalid MNIST labels header"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_labels_magic(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 1
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * 784))
    # Wrong magic (2048)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n, magic=2048) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_count_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # n mismatch between image header and labels header (2 vs 1)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(2) + bytes([0] * (2 * 784)))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_truncated_labels(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 3
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * (n * 784)))
    # Only n-1 labels
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n) + bytes([0] * (n - 1)))
    with pytest.raises(RuntimeError, match="truncated MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_build_dataset_from_spec_mnist(tmp_path: Path) -> None:
    # Create valid raw MNIST files
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 4
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * (n * 784)))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n) + bytes([i % 10 for i in range(n)]))

    spec = PreprocessSpec(
        base_kind="mnist",
        mnist=MNISTSpec(root=tmp_path, train=True),
        inline=None,
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds = _build_dataset_from_spec(spec)
    assert isinstance(ds, PreprocessDataset)
    assert len(ds) == n

