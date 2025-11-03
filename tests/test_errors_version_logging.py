from __future__ import annotations

import io
import logging

import pytest

from handwriting_ai.errors import ErrorCode, new_error, status_for
from handwriting_ai.logging import _JsonFormatter, get_logger, log_event
from handwriting_ai.version import get_version


def test_timeout_status_and_new_error_default_message() -> None:
    assert status_for(ErrorCode.timeout) == 504
    rid = "abc-123"
    e = new_error(ErrorCode.timeout, rid)
    assert e.message != "" and e.request_id == rid


def test_version_fallback_logs_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    class _Mock:
        def __init__(self) -> None:
            self.exc = Exception()

        def version(self, _: str) -> str:
            raise self.PackageNotFoundError()

        class PackageNotFoundError(Exception):  # PackageNotFoundError replacement
            pass

    # Monkeypatch module to raise PNF
    meta = importlib.import_module("importlib.metadata")
    monkeypatch.setattr(meta, "version", _Mock().version, raising=False)
    monkeypatch.setattr(meta, "PackageNotFoundError", _Mock.PackageNotFoundError, raising=False)

    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        v = get_version()
    finally:
        logger.removeHandler(h)
    assert v.version == "0.1.0"
    assert "pkg_version_fallback" in buf.getvalue()


def test_log_event_structured_types() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        log_event(
            "custom",
            fields={
                "latency_ms": 7,
                "digit": 3,
                "confidence": 0.25,
                "model_id": "m",
                "uncertain": False,
            },
        )
    finally:
        logger.removeHandler(h)
    out = buf.getvalue()
    assert '"message": "custom"' in out
    assert '"latency_ms": 7' in out and '"digit": 3' in out
    assert '"confidence": 0.25' in out and '"uncertain": false' in out
