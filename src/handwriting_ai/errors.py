from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from fastapi import status


class ErrorCode(str, Enum):
    invalid_image = "invalid_image"
    unsupported_media_type = "unsupported_media_type"
    bad_dimensions = "bad_dimensions"
    too_large = "too_large"
    preprocessing_failed = "preprocessing_failed"
    timeout = "timeout"
    internal_error = "internal_error"
    unauthorized = "unauthorized"
    malformed_multipart = "malformed_multipart"
    service_not_ready = "service_not_ready"


_DEFAULT_MESSAGE: Final[dict[ErrorCode, str]] = {
    ErrorCode.invalid_image: "Failed to decode image.",
    ErrorCode.unsupported_media_type: "Unsupported media type.",
    ErrorCode.bad_dimensions: "Image dimensions exceed allowed limits.",
    ErrorCode.too_large: "File exceeds size limit.",
    ErrorCode.preprocessing_failed: "Image preprocessing failed.",
    ErrorCode.timeout: "Request timed out.",
    ErrorCode.internal_error: "Internal server error.",
    ErrorCode.unauthorized: "Unauthorized.",
    ErrorCode.malformed_multipart: "Malformed multipart body.",
    ErrorCode.service_not_ready: "Model not loaded. Upload or train a model.",
}


@dataclass(frozen=True)
class ErrorResponse:
    code: ErrorCode
    message: str
    request_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "message": self.message,
            "request_id": self.request_id,
        }


class AppError(Exception):
    def __init__(self, code: ErrorCode, http_status: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.message = message


def new_error(code: ErrorCode, request_id: str, message: str | None = None) -> ErrorResponse:
    msg = message if message is not None else _DEFAULT_MESSAGE.get(code, "")
    return ErrorResponse(code=code, message=msg, request_id=request_id)


def status_for(code: ErrorCode) -> int:
    if code is ErrorCode.invalid_image:
        return status.HTTP_400_BAD_REQUEST
    if code is ErrorCode.unsupported_media_type:
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    if code is ErrorCode.bad_dimensions:
        return status.HTTP_400_BAD_REQUEST
    if code is ErrorCode.too_large:
        return status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    if code is ErrorCode.preprocessing_failed:
        return status.HTTP_400_BAD_REQUEST
    if code is ErrorCode.timeout:
        # Use 500 if internal timeout boundary triggered; endpoint may map differently.
        return status.HTTP_504_GATEWAY_TIMEOUT
    if code is ErrorCode.unauthorized:
        return status.HTTP_401_UNAUTHORIZED
    if code is ErrorCode.malformed_multipart:
        return status.HTTP_400_BAD_REQUEST
    if code is ErrorCode.service_not_ready:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_500_INTERNAL_SERVER_ERROR
