from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from handwriting_ai.api.app import create_app


def _png_bytes() -> bytes:
    img = Image.new("L", (28, 28), 0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_read_returns_service_not_ready_when_model_missing(tmp_path: Path) -> None:
    app = create_app()
    client = TestClient(app)
    files = {"file": ("d.png", _png_bytes(), "image/png")}
    resp = client.post("/v1/read", files=files)
    assert resp.status_code == 503
    body_obj: object = resp.json()
    assert isinstance(body_obj, dict)
    body: dict[str, object] = body_obj
    assert body.get("code") == "service_not_ready"
    msg_val = body.get("message")
    assert isinstance(msg_val, str) and ("Model not loaded" in msg_val)
    rid = body.get("request_id")
    assert isinstance(rid, str) and len(rid) > 0
