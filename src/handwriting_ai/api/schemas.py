from __future__ import annotations

from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True)
class PredictResponse:
    digit: int
    confidence: float
    probs: list[float]
    model_id: str
    visual_png_b64: str | None
    uncertain: bool
    latency_ms: int
