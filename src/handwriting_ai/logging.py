from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final

from .request_context import request_id_var

_LOGGER_NAME: Final[str] = "handwriting_ai"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": rid,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def init_logging() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if called multiple times (tests/process reuse)
    if not _has_stream_handler(logger):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonFormatter())
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def _has_stream_handler(logger: logging.Logger) -> bool:
    # Avoid the built-in helper that checks a condition across iterables
    return next(
        (True for h in logger.handlers if isinstance(h, logging.StreamHandler)),
        False,
    )
