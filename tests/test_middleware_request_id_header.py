from __future__ import annotations

from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def test_request_id_header_roundtrip() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s)
    client = TestClient(app)
    r = client.get("/healthz", headers={"X-Request-ID": "req-123"})
    # Middleware should propagate request id to response.
    # Use string form to avoid type ambiguity.
    hs = str(r.headers)
    assert "x-request-id" in hs.lower() and "req-123" in hs
