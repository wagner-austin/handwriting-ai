from __future__ import annotations

import io
import logging

from handwriting_ai.logging import _JsonFormatter, get_logger, log_event


def test_log_event_ignores_wrong_field_types() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        # Provide wrong types for all optional fields
        log_event(
            "custom",
            fields={
                "latency_ms": 1.2,  # not int
                "digit": 3.4,  # not int
                "confidence": "0.5",  # not float
                "model_id": 123,  # not str
                "uncertain": "no",  # not bool
            },
        )
    finally:
        logger.removeHandler(h)
    s = buf.getvalue()
    # Only message should be present; optional fields skipped
    assert '"message": "custom"' in s
    assert "latency_ms" not in s and "digit" not in s and "confidence" not in s
    assert "model_id" not in s and "uncertain" not in s
