from __future__ import annotations

import pytest
from PIL import Image

from handwriting_ai.preprocess import (
    _estimate_background_is_dark,
    _largest_component_crop,
    _otsu_binarize,
)


def test_estimate_background_median_loop_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force histogram with uniform counts to exercise multiple loop iterations
    def _hist(self: Image.Image) -> list[int]:
        return [1] * 256

    monkeypatch.setattr(Image.Image, "histogram", _hist, raising=True)
    img = Image.new("L", (4, 4), 127)
    out = _estimate_background_is_dark(img)
    assert isinstance(out, bool)


def test_otsu_binarize_various_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Histogram: some dark + some bright to ensure var update and continue path
    hist = [0] * 256
    hist[10] = 10
    hist[200] = 5

    def _hist(self: Image.Image) -> list[int]:
        return hist

    monkeypatch.setattr(Image.Image, "histogram", _hist, raising=True)
    img = Image.new("L", (4, 4), 0)
    bw = _otsu_binarize(img)
    assert bw.mode == "L"


def test_estimate_background_immediate_break(monkeypatch: pytest.MonkeyPatch) -> None:
    # Heavy first bin forces break at first iteration
    def _hist(self: Image.Image) -> list[int]:
        h = [0] * 256
        h[0] = 20
        h[1] = 1
        return h

    monkeypatch.setattr(Image.Image, "histogram", _hist, raising=True)
    img = Image.new("L", (4, 4), 0)
    _ = _estimate_background_is_dark(img)


def test_otsu_binarize_no_continue_at_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-zero first bin means w_b != 0 on first iteration, covering that branch
    def _hist(self: Image.Image) -> list[int]:
        h = [0] * 256
        h[0] = 5
        h[200] = 5
        return h

    monkeypatch.setattr(Image.Image, "histogram", _hist, raising=True)
    img = Image.new("L", (4, 4), 0)
    _ = _otsu_binarize(img)


def test_largest_component_equal_area_false_branch() -> None:
    # Two equal 1-pixel components ensure area>best_area false path executes
    bw = Image.new("L", (4, 4), 0)
    bw.putpixel((0, 0), 255)
    bw.putpixel((3, 3), 255)
    out = _largest_component_crop(bw)
    assert out.size[0] >= 1 and out.size[1] >= 1
