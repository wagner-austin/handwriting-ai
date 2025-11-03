from __future__ import annotations

import uuid
from collections.abc import Callable

from fastapi import Header
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from .config import Settings
from .errors import AppError, ErrorCode, status_for
from .request_context import request_id_var


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("X-Request-ID")
        if not rid:
            rid = str(uuid.uuid4())
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response


def api_key_dependency(settings: Settings) -> Callable[[str | None], None]:
    required_key = settings.security.api_key.strip()
    if required_key == "":
        # No-op dependency when key is not configured; keep header signature optional
        def _pass(x_api_key: str | None = Header(default=None, convert_underscores=True)) -> None:
            return None

        return _pass

    def _check(x_api_key: str | None = Header(default=None, convert_underscores=True)) -> None:
        if x_api_key is None or x_api_key != required_key:
            raise AppError(
                ErrorCode.unauthorized,
                status_for(ErrorCode.unauthorized),
                "Unauthorized",
            )

    return _check
