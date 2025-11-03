from __future__ import annotations

from PIL import Image

from handwriting_ai.preprocess import (
    _center_on_square,
    _deskew_if_needed,
    _otsu_binarize,
    _principal_angle,
)


def test_principal_angle_none_when_no_pixels() -> None:
    img = Image.new("L", (32, 32), 255)
    ang = _principal_angle(img, 32, 32)
    assert ang is None


def test_deskew_bbox_none_fallback() -> None:
    img = Image.new("L", (32, 32), 255)
    out = _deskew_if_needed(img)
    assert out.size == img.size


def test_otsu_uniform_grayscale() -> None:
    img = Image.new("L", (16, 16), 127)
    bw = _otsu_binarize(img)
    assert bw.mode == "L" and bw.size == img.size


def test_center_on_square_basic() -> None:
    img = Image.new("L", (10, 6), 255)
    for x in range(3, 7):
        for y in range(2, 5):
            img.putpixel((x, y), 0)
    out = _center_on_square(img)
    assert out.size[0] == out.size[1]
