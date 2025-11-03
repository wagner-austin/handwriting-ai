from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def _mk_png_bytes() -> bytes:
    img = Image.new("L", (8, 8), 255)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_rejects_extra_form_field() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s)
    client = TestClient(app)
    files = {"file": ("img.png", _mk_png_bytes(), "image/png")}
    data = {"note": "extra"}
    r = client.post("/v1/read", files=files, data=data)
    assert r.status_code == 400 and '"code":"malformed_multipart"' in r.text


def test_rejects_multiple_file_parts() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s)
    client = TestClient(app)
    files_list = [
        ("file", ("a.png", _mk_png_bytes(), "image/png")),
        ("file", ("b.png", _mk_png_bytes(), "image/png")),
    ]
    r = client.post("/v1/read", files=files_list)
    assert r.status_code == 400 and '"code":"malformed_multipart"' in r.text
