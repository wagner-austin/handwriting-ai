from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def _mk_png_bytes() -> bytes:
    img = Image.new("L", (64, 64), 255)
    for y in range(20, 44):
        for x in range(28, 36):
            img.putpixel((x, y), 0)
    b = BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def test_routes_health_ready_version_and_read() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s)
    client = TestClient(app)

    r1 = client.get("/healthz")
    assert r1.status_code == 200
    assert '"status":"ok"' in r1.text

    r2 = client.get("/readyz")
    assert r2.status_code == 200
    assert "status" in r2.text

    r3 = client.get("/version")
    assert r3.status_code == 200
    assert "handwriting-ai" in r3.text

    # Unsupported media type
    r5 = client.post("/v1/read", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r5.status_code == 415
