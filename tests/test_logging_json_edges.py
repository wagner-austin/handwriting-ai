from __future__ import annotations

import io
import logging

from handwriting_ai.logging import _JsonFormatter, get_logger, log_event


def test_json_formatter_evt_without_event_field() -> None:
    rec = logging.LogRecord(
        name="handwriting_ai",
        level=logging.INFO,
        pathname="t",
        lineno=1,
        msg="EVT foo=1",
        args=(),
        exc_info=None,
    )
    out = _JsonFormatter().format(rec)
    # Without an explicit event= token, message remains unchanged and extra fields are added
    assert '"message": "EVT foo=1"' in out
    assert '"foo": "1"' in out


def test_log_event_with_none_fields_produces_only_message() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        log_event("abc", fields=None)
    finally:
        logger.removeHandler(h)
    s = buf.getvalue()
    assert '"message": "abc"' in s and "latency_ms" not in s and "digit" not in s
