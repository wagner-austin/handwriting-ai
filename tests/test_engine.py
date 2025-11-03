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


def test_engine_tta_batching() -> None:
    # Dummy model that records batch size and returns zero logits
    class _M:
        def __init__(self) -> None:
            self.last_batch: int = 0

        def eval(self) -> object:
            return self

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            self.last_batch = int(x.shape[0]) if x.ndim == 4 else 1
            return torch.zeros((self.last_batch, 10), dtype=torch.float32)

        def load_state_dict(self, sd: dict[str, torch.Tensor]) -> object:
            return self

    def _mk_engine(tta: bool) -> InferenceEngine:
        s = Settings(app=AppConfig(), digits=DigitsConfig(tta=tta), security=SecurityConfig())
        eng = InferenceEngine(s)
        # Inject manifest and model directly
        from datetime import UTC, datetime

        from handwriting_ai.inference.manifest import ModelManifest

        eng._manifest = ModelManifest(
            schema_version="v1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.0,
            temperature=1.0,
        )
        eng._model = _M()  # assign dummy model implementing protocol
        return eng

    e0 = _mk_engine(tta=False)
    _ = e0._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert isinstance(e0._model, _M)
    assert e0._model.last_batch == 1

    e1 = _mk_engine(tta=True)
    _ = e1._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert isinstance(e1._model, _M)
    assert e1._model.last_batch > 1
