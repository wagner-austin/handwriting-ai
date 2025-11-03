from __future__ import annotations

import pytest
from PIL import Image

from handwriting_ai.preprocess import _deskew_if_needed


def test_deskew_confidence_gate_skips_when_low(monkeypatch: pytest.MonkeyPatch) -> None:
    img = Image.new("L", (10, 10), 255)
    from handwriting_ai import preprocess as _pp

    def _fake(_: Image.Image, __: int, ___: int) -> tuple[float, float] | None:
        return (5.0, 0.0)

    monkeypatch.setattr(_pp, "_principal_angle_confidence", _fake, raising=True)
    out = _deskew_if_needed(img)
    assert out is img


def test_deskew_small_angle_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    img = Image.new("L", (10, 10), 255)
    from handwriting_ai import preprocess as _pp

    def _fake(_: Image.Image, __: int, ___: int) -> tuple[float, float] | None:
        return (0.5, 1.0)

    monkeypatch.setattr(_pp, "_principal_angle_confidence", _fake, raising=True)
    out = _deskew_if_needed(img)
    assert out is img


def test_deskew_negative_angle_clamps_and_bbox_none_returns_img(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    img = Image.new("L", (10, 10), 255)
    from handwriting_ai import preprocess as _pp

    def _fake(_: Image.Image, __: int, ___: int) -> tuple[float, float] | None:
        return (-20.0, 1.0)

    monkeypatch.setattr(_pp, "_principal_angle_confidence", _fake, raising=True)
    out = _deskew_if_needed(img)
    assert out is img
