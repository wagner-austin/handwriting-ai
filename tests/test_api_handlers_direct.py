from __future__ import annotations

import anyio
from starlette.requests import Request
from starlette.types import Scope

from handwriting_ai.api.app import _handle_app_error, _handle_unexpected


def test_handle_app_error_with_non_app_error() -> None:
    async def run() -> None:
        scope: Scope = {"type": "http"}
        req = Request(scope)
        resp = await _handle_app_error(req, Exception("boom"))
        raw = resp.body
        body = (raw if isinstance(raw, bytes | bytearray) else bytes(raw)).decode("utf-8")
        assert '"code":"internal_error"' in body and '"request_id"' in body

    anyio.run(run)


def test_handle_unexpected_generic_exception() -> None:
    async def run() -> None:
        scope: Scope = {"type": "http"}
        req = Request(scope)
        resp = await _handle_unexpected(req, Exception("boom"))
        raw = resp.body
        body = (raw if isinstance(raw, bytes | bytearray) else bytes(raw)).decode("utf-8")
        assert '"code":"internal_error"' in body and '"request_id"' in body

    anyio.run(run)
