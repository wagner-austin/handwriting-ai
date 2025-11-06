from __future__ import annotations

from PIL import Image

from handwriting_ai.preprocess import (
    _estimate_background_is_dark,
    _largest_component_crop,
    _otsu_binarize,
)


def test_background_white_returns_false() -> None:
    img = Image.new("L", (8, 8), 255)
    assert _estimate_background_is_dark(img) is False


def test_largest_component_crop_compares_areas() -> None:
    # Two components: a 1x1 pixel and a 2x1 block
    bw = Image.new("L", (4, 4), 0)
    bw.putpixel((0, 0), 255)
    bw.putpixel((2, 2), 255)
    bw.putpixel((3, 2), 255)
    out = _largest_component_crop(bw)
    # Expect the 2x1 component crop (width >= 2)
    assert out.size[0] >= 2 and out.size[1] >= 1


def test_otsu_all_white_triggers_break() -> None:
    img = Image.new("L", (4, 4), 255)
    bw = _otsu_binarize(img)
    # All pixels remain white
    assert all(bw.getpixel((x, y)) == 255 for y in range(4) for x in range(4))
