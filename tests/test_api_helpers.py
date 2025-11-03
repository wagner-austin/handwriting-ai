from __future__ import annotations

import pytest

from handwriting_ai.api.app import _raise_if_too_large
from handwriting_ai.config import AppConfig, DigitsConfig, Limits, SecurityConfig, Settings
from handwriting_ai.errors import AppError, ErrorCode


def test_raise_if_too_large_raises_with_oversize() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(max_image_mb=0), security=SecurityConfig())
    limits = Limits.from_settings(s)
    with pytest.raises(AppError) as ei:
        _raise_if_too_large(b"x" * 1024, limits)
    e = ei.value
    assert e.code is ErrorCode.too_large and e.http_status == 413


def test_raise_if_too_large_ok_within_limit() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(max_image_mb=1), security=SecurityConfig())
    limits = Limits.from_settings(s)
    # 1 byte is well below 1MB
    _raise_if_too_large(b"x", limits)
