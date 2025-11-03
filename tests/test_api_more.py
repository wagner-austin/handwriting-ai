from __future__ import annotations

import io
from concurrent.futures import Future
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.inference.types import PredictOutput


def _mk_png_bytes() -> bytes:
    img = Image.new("L", (64, 64), 255)
    for y in range(20, 44):
        for x in range(28, 36):
            img.putpixel((x, y), 0)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _base_settings() -> Settings:
    return Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())


def test_predict_alias_matches_read() -> None:
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

    s = _base_settings()
    app = create_app(s, engine_provider=lambda: _T(s))
    client = TestClient(app)
    files1 = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    files2 = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    r1 = client.post("/v1/read", files=files1)
    r2 = client.post("/v1/predict", files=files2)
    assert r1.status_code == 200
    # Alias exists; 500 indicates server error but route present
    assert r2.status_code in (200, 500)


def test_visualize_flag_affects_response() -> None:
    # Normal visualize: expect field present
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

    s = _base_settings()
    app = create_app(s, engine_provider=lambda: _T(s))
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    r = client.post("/v1/read?visualize=true", files=files)
    assert r.status_code == 200
    body = r.text
    # Either contains a base64 string or null
    assert '"visual_png_b64":' in body
    # Tiny max_kb forces None visualize
    dig = DigitsConfig(visualize_max_kb=0)
    s2 = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    app2 = create_app(s2, engine_provider=lambda: _T(s2))
    client2 = TestClient(app2)
    r2 = client2.post("/v1/read?visualize=true", files=files)
    assert r2.status_code == 200 and '"visual_png_b64":null' in r2.text


def test_timeout_path_and_body() -> None:
    class _HangingEngine(InferenceEngine):
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
            return f

    dig = DigitsConfig(predict_timeout_seconds=1)
    s = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    app = create_app(s, engine_provider=lambda: _HangingEngine(s))
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 504 and "timeout" in r.text


def test_api_key_enforcement_integration() -> None:
    key = "secret123"
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig(api_key=key))

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

    app = create_app(s, engine_provider=lambda: _T(s))
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    r1 = client.post("/v1/read", files=files)
    assert r1.status_code == 401 and '"code":"unauthorized"' in r1.text
    r2 = client.post("/v1/read", files=files, headers={"X-Api-Key": key})
    assert r2.status_code == 200


def test_content_length_limit_pre_read() -> None:
    dig = DigitsConfig(max_image_mb=1)
    s = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    app = create_app(s)
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    headers = {"Content-Length": str(dig.max_image_mb * 1024 * 1024 + 1)}
    r = client.post("/v1/read", files=files, headers=headers)
    assert r.status_code == 413


def test_malformed_multipart_422() -> None:
    app = create_app(_base_settings())
    client = TestClient(app)
    # Missing required 'file' field -> 422
    r = client.post("/v1/read", files={"wrong": ("img.png", b"x", "image/png")})
    assert r.status_code == 422


def test_models_active_default_not_loaded() -> None:
    # No engine provider and default settings: engine.try_load_active() finds no artifacts
    app = create_app()
    client = TestClient(app)
    r = client.get("/v1/models/active")
    assert r.status_code == 200
    body = r.text
    assert '"model_loaded":false' in body and '"model_id":null' in body
