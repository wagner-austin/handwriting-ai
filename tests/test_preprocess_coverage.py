from __future__ import annotations

import pytest
from PIL import Image, ImageOps

from handwriting_ai.errors import AppError, ErrorCode
from handwriting_ai.preprocess import (
    PreprocessOptions,
    _center_on_square,
    _component_bbox_bytes,
    _deskew_if_needed,
    _estimate_background_is_dark,
    _largest_component_crop,
    _principal_angle,
    run_preprocess,
)


def _mk_white(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 255)


def _mk_black(size: tuple[int, int]) -> Image.Image:
    return Image.new("L", size, 0)


def test_run_preprocess_catches_non_app_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((8, 8))

    def _boom(_: Image.Image) -> Image.Image:
        raise TypeError("bad exif")

    monkeypatch.setattr(ImageOps, "exif_transpose", _boom, raising=True)
    opts = PreprocessOptions(invert=None, center=True, visualize=False, visualize_max_kb=1)
    with pytest.raises(AppError) as ei:
        _ = run_preprocess(img, opts)
    e = ei.value
    assert e.code is ErrorCode.preprocessing_failed and e.http_status == 400


def test_load_to_grayscale_exif_none(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((8, 8))

    def _none(_: Image.Image) -> Image.Image | None:  # returns None to trigger path
        return None

    monkeypatch.setattr(ImageOps, "exif_transpose", _none, raising=True)
    opts = PreprocessOptions(invert=None, center=True, visualize=False, visualize_max_kb=1)
    with pytest.raises(AppError) as ei:
        _ = run_preprocess(img, opts)
    assert ei.value.code is ErrorCode.invalid_image


def test_background_total_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((4, 4))

    def _zeros(self: Image.Image) -> list[int]:
        return [0] * 256

    monkeypatch.setattr(Image.Image, "histogram", _zeros, raising=True)
    assert _estimate_background_is_dark(img) is False


def test_largest_component_buffer_size_mismatch_raises() -> None:
    rgb = Image.new("RGB", (3, 3), (0, 0, 0))
    with pytest.raises(AppError) as ei:
        _ = _largest_component_crop(rgb.convert("RGB"))
    e = ei.value
    assert e.code is ErrorCode.preprocessing_failed and "buffer size" in e.message


def test_component_bbox_updates_minx_miny() -> None:
    width, height = 3, 3
    # Build a small component including (0,0), (1,0), (0,1); start at (1,1)
    vals: list[int] = [0] * (width * height)

    def idx(x: int, y: int) -> int:
        return y * width + x

    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        vals[idx(x, y)] = 255
    buf = bytes(vals)
    visited = [False] * (width * height)
    area, bbox = _component_bbox_bytes(buf, visited, width, height, 1, 1)
    x0, y0, x1, y1 = bbox
    assert area >= 4 and x0 == 0 and y0 == 0 and x1 >= 1 and y1 >= 1


def test_deskew_negative_angle_clamp_and_bbox_none(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((10, 10))

    def _angle(_: Image.Image, __: int, ___: int) -> float:
        return -20.0

    from handwriting_ai import preprocess as _pp

    monkeypatch.setattr(_pp, "_principal_angle", _angle, raising=True)
    out = _deskew_if_needed(img)
    # No content so bbox is None; function returns original image
    assert out is img


def test_principal_angle_none_when_pix_none(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((3, 3))

    def _load_none(self: Image.Image) -> None:
        return None

    monkeypatch.setattr(Image.Image, "load", _load_none, raising=True)
    assert _principal_angle(img, 3, 3) is None


def test_principal_angle_none_when_zero_variance() -> None:
    img = _mk_white((4, 4))
    img.putpixel((2, 2), 0)
    assert _principal_angle(img, 4, 4) is None


def test_center_on_square_pix_none_returns_input(monkeypatch: pytest.MonkeyPatch) -> None:
    img = _mk_white((5, 5))

    def _load_none(self: Image.Image) -> None:
        return None

    monkeypatch.setattr(Image.Image, "load", _load_none, raising=True)
    out = _center_on_square(img)
    assert out is img


def test_center_on_square_no_pixels_returns_input() -> None:
    img = _mk_black((6, 6))
    out = _center_on_square(img)
    assert out is img
