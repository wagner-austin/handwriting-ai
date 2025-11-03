from __future__ import annotations

import torch

from handwriting_ai.inference.engine import _augment_for_tta


def test_augment_for_tta_includes_rotations() -> None:
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    out = _augment_for_tta(x)
    # Expect identity + 4 shifts + 4 rotations = 9
    assert tuple(out.shape) == (9, 1, 28, 28)
