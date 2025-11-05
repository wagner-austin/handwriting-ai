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
from handwriting_ai.preprocess import preprocess_signature


def _png_bytes() -> bytes:
    img = Image.new("L", (28, 28), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _ErrEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Provide a manifest so service_not_ready is not triggered
        self._manifest = ModelManifest(
            schema_version="v1.1",
            model_id="m",
            arch="resnet18",
            n_classes=10,
            version="1.0.0",
            created_at=datetime.now(UTC),
            preprocess_hash=preprocess_signature(),
            val_acc=0.0,
            temperature=1.0,
        )

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        f: Future[PredictOutput] = Future()
        f.set_exception(RuntimeError("boom"))
        return f


def test_predict_internal_error_500() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s, engine_provider=lambda: _ErrEngine(s))
    client = TestClient(app, raise_server_exceptions=False)
    files = {"file": ("img.png", _png_bytes(), "image/png")}
    r = client.post("/v1/read", files=files)
    assert r.status_code == 500
    body_obj: object = r.json()
    assert isinstance(body_obj, dict)
    body: dict[str, object] = body_obj
    assert body.get("code") == "internal_error"
    assert isinstance(body.get("request_id"), str)
