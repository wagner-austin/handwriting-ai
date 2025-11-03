from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PredictOutput:
    digit: int
    confidence: float
    probs: tuple[float, ...]  # length 10
    model_id: str


@dataclass(frozen=True)
class PreprocessOutput:
    tensor: Tensor
    visual_png: bytes | None


Probs = Sequence[float]
