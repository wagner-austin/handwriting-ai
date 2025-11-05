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
    invalid_model = "invalid_model"


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
    ErrorCode.invalid_model: "Invalid model file.",
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
    mapping: dict[ErrorCode, int] = {
        ErrorCode.invalid_image: status.HTTP_400_BAD_REQUEST,
        ErrorCode.unsupported_media_type: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ErrorCode.bad_dimensions: status.HTTP_400_BAD_REQUEST,
        ErrorCode.too_large: status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        ErrorCode.preprocessing_failed: status.HTTP_400_BAD_REQUEST,
        ErrorCode.timeout: status.HTTP_504_GATEWAY_TIMEOUT,
        ErrorCode.unauthorized: status.HTTP_401_UNAUTHORIZED,
        ErrorCode.malformed_multipart: status.HTTP_400_BAD_REQUEST,
        ErrorCode.service_not_ready: status.HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.invalid_model: status.HTTP_400_BAD_REQUEST,
    }
    return mapping.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)
