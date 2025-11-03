from __future__ import annotations

import io
import logging

from handwriting_ai.logging import _JsonFormatter, get_logger


def test_exc_info_present_in_json() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("oops")
    finally:
        logger.removeHandler(h)
    out = buf.getvalue()
    assert '"exc_info":' in out and "Traceback" in out


def test_evt_parser_ignores_non_evt_messages() -> None:
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(_JsonFormatter())
    logger = get_logger()
    logger.addHandler(h)
    try:
        logger.info("hello world")
    finally:
        logger.removeHandler(h)
    out = buf.getvalue()
    assert '"message": "hello world"' in out and "latency_ms" not in out
