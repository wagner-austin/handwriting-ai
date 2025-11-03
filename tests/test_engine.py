from __future__ import annotations

import torch

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine


def test_engine_submit_with_zero_model() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    eng = InferenceEngine(s)
    # Engine without a loaded model should reject predict.
    # Calling the internal implementation raises RuntimeError.
    raised = False
    try:
        _ = eng._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    except RuntimeError:
        raised = True
    assert raised is True
