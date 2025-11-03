from __future__ import annotations

from concurrent.futures import Future
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from handwriting_ai.api.app import create_app
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
            val_acc=0.0,
            temperature=1.0,
        )

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        probs = tuple(0.1 for _ in range(10))
        f.set_result(PredictOutput(digit=0, confidence=0.5, probs=probs, model_id="test_model"))
        return f


def _base_app() -> TestClient:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s, engine_provider=lambda: _T(s))
    return TestClient(app)


def test_invalid_image_bytes_returns_400() -> None:
    client = _base_app()
    files = {"file": ("img.png", b"not-a-valid-image", "image/png")}
    # Force header pre-check to be bypassed by sending Content-Length=0
    r = client.post("/v1/read", files=files, headers={"Content-Length": "0"})
    assert r.status_code == 400 and "invalid_image" in r.text


def test_decompression_bomb_error_returns_413(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _base_app()

    def _raise_bomb(_: object) -> Image.Image:
        raise Image.DecompressionBombError("bomb")

    monkeypatch.setattr(Image, "open", _raise_bomb, raising=True)
    files = {"file": ("img.png", b"header-wont-be-read", "image/png")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 413 and "Decompression bomb" in r.text


def test_post_read_size_limit_returns_413() -> None:
    # Configure extremely low size to trigger post-read limit check
    dig = DigitsConfig(max_image_mb=0)
    s = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    app = create_app(s, engine_provider=lambda: _T(s))
    client = TestClient(app)
    files = {"file": ("img.png", b"x" * 2048, "image/png")}
    r = client.post("/v1/read", files=files)
    # Accept either pre-read or post-read too_large messages
    assert r.status_code == 413 and '"code":"too_large"' in r.text
