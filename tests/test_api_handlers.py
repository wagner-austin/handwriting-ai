from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.errors import AppError, ErrorCode


def _mk_app() -> FastAPI:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s)

    # Add dynamic routes to trigger handlers
    def _r1() -> None:
        raise AppError(ErrorCode.invalid_image, 400, "bad")

    app.add_api_route("/raise-app-error", _r1, methods=["GET"])

    def _r2() -> None:
        raise RuntimeError("boom")

    app.add_api_route("/raise-exception", _r2, methods=["GET"])

    return app


def test_app_error_handler_shapes_body() -> None:
    app = _mk_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-app-error")
    assert r.status_code == 400
    assert '"code":"invalid_image"' in r.text and '"request_id"' in r.text


def test_unexpected_handler_shapes_body() -> None:
    app = _mk_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/raise-exception")
    assert r.status_code == 500
    assert '"code":"internal_error"' in r.text and '"request_id"' in r.text
