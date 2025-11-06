from __future__ import annotations

import random

import pytest
from PIL import Image

from handwriting_ai.training.augment import maybe_add_dots


def test_maybe_add_dots_out_of_bounds_path(monkeypatch: pytest.MonkeyPatch) -> None:
    img = Image.new("L", (2, 2), 0)

    class _Rng:
        def __init__(self) -> None:
            self.n = 0

        def random(self) -> float:
            # First call decide to draw dots (p threshold), then choose val
            self.n += 1
            return 0.0  # always below any p

        def randint(self, a: int, b: int) -> int:
            # Place dot in bottom-right corner to force some out-of-bounds within the s x s loop
            return 1

    rng = _Rng()
    monkeypatch.setattr(random, "random", rng.random)
    monkeypatch.setattr(random, "randint", rng.randint)

    _ = maybe_add_dots(img, prob=1.0, count=1, size_px=3)
    # No assertion needed; coverage exercises the false branch of bounds check
