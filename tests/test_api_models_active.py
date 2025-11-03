from __future__ import annotations

from concurrent.futures import Future
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from torch import Tensor

from handwriting_ai.api.app import _register_models
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput


class _T(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._manifest = ModelManifest(
            schema_version="v1",
            model_id="test_model",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
            val_acc=0.75,
            temperature=1.0,
        )

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        probs = tuple(0.1 for _ in range(10))
        f.set_result(PredictOutput(digit=0, confidence=0.5, probs=probs, model_id="test_model"))
        return f


def test_models_active_ready_path_returns_fields() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    eng = _T(s)
    eng._manifest = ModelManifest(
        schema_version="v1",
        model_id="test_model",
        arch="resnet18",
        n_classes=10,
        version="1.0.0",
        created_at=datetime.now(UTC),
        preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
        val_acc=0.75,
        temperature=1.0,
    )
    app = FastAPI()
    _register_models(app, eng)
    client = TestClient(app)
    r = client.get("/v1/models/active")
    assert r.status_code == 200
    body = r.text
    assert '"model_loaded":true' in body
    assert '"model_id":"test_model"' in body
    assert '"arch":"resnet18"' in body
    assert '"n_classes":10' in body
    assert '"val_acc":0.75' in body
    assert '"temperature":1.0' in body
