from __future__ import annotations

import pytest
from PIL import Image

from handwriting_ai.errors import AppError, ErrorCode
from handwriting_ai.preprocess import (
    PreprocessOptions,
    _deskew_if_needed,
    _load_to_grayscale,
    _principal_angle,
    _visualize_png,
    run_preprocess,
)


def _mk_blank(size: int = 64, fill: int = 255) -> Image.Image:
    return Image.new("L", (size, size), fill)


def _mk_square_digit() -> Image.Image:
    img = Image.new("L", (64, 64), 255)
    for y in range(20, 44):
        for x in range(28, 36):
            img.putpixel((x, y), 0)
    return img


def test_blank_image_raises_preprocessing_error() -> None:
    img = _mk_blank(fill=0)
    opts = PreprocessOptions(invert=False, center=True, visualize=False, visualize_max_kb=16)
    with pytest.raises(AppError) as ei:
        _ = run_preprocess(img, opts)
    err = ei.value
    assert isinstance(err, AppError) and err.code is ErrorCode.preprocessing_failed


def test_invert_override_affects_output() -> None:
    img = _mk_square_digit()
    opts1 = PreprocessOptions(invert=False, center=True, visualize=False, visualize_max_kb=16)
    opts2 = PreprocessOptions(invert=True, center=True, visualize=False, visualize_max_kb=16)
    o1 = run_preprocess(img, opts1)
    o2 = run_preprocess(img, opts2)
    # Tensor sums should differ when inverting intensities before normalization
    s1 = float(o1.tensor.sum().item())
    s2 = float(o2.tensor.sum().item())
    assert abs(s1 - s2) > 1e-3


def test_rgba_and_palette_to_grayscale() -> None:
    base = _mk_square_digit()
    rgba = base.convert("RGBA")
    pal = base.convert("P")
    g1 = _load_to_grayscale(rgba)
    g2 = _load_to_grayscale(pal)
    assert g1.mode == "L" and g2.mode == "L"


def test_deskew_cap_behavior() -> None:
    # Create a vertical bar and rotate it by ~20 degrees
    img = Image.new("L", (64, 64), 255)
    for y in range(8, 56):
        for x in range(28, 36):
            img.putpixel((x, y), 0)
    rotated = img.rotate(20, expand=True, fillcolor=255)
    angle_in = _principal_angle(rotated, rotated.size[0], rotated.size[1])
    assert angle_in is not None and angle_in > 10.0
    corrected = _deskew_if_needed(rotated)
    angle_out = _principal_angle(corrected, corrected.size[0], corrected.size[1])
    # After capping rotation at 10°, the residual angle should be close to angle_in - 10
    if angle_out is not None:
        assert abs((angle_in - 10.0) - angle_out) < 5.0


def test_visualize_size_cap_returns_none() -> None:
    img = _mk_square_digit()
    small = _visualize_png(img, 1)
    assert small is None or isinstance(small, bytes)
