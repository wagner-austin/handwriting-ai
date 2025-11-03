from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final, TypedDict

from .request_context import request_id_var

_LOGGER_NAME: Final[str] = "handwriting_ai"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": rid,
        }
        # Parse structured fields encoded via log_event helper
        msg = record.getMessage()
        extra = _parse_evt_fields(msg)
        if extra:
            if "event" in extra:
                payload["message"] = str(extra.pop("event"))
            for k, v in extra.items():
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class LogEvent(TypedDict, total=False):
    event: str
    latency_ms: int
    digit: int
    confidence: float
    model_id: str
    uncertain: bool


def log_event(event: str, fields: LogEvent | None = None) -> None:
    logger = get_logger()
    parts: list[str] = [f"event={event}"]
    if fields is not None:
        if "latency_ms" in fields and isinstance(fields["latency_ms"], int):
            parts.append(f"latency_ms={fields['latency_ms']}")
        if "digit" in fields and isinstance(fields["digit"], int):
            parts.append(f"digit={fields['digit']}")
        if "confidence" in fields and isinstance(fields["confidence"], float):
            parts.append(f"confidence={fields['confidence']}")
        if "model_id" in fields and isinstance(fields["model_id"], str):
            # Avoid spaces in value
            parts.append(f"model_id={fields['model_id']}")
        if "uncertain" in fields and isinstance(fields["uncertain"], bool):
            parts.append(f"uncertain={'true' if fields['uncertain'] else 'false'}")
    logger.info("EVT " + " ".join(parts))


def _parse_evt_fields(msg: str) -> dict[str, object]:
    if not isinstance(msg, str) or not msg.startswith("EVT "):
        return {}
    out: dict[str, object] = {}
    body = msg[4:]
    for tok in body.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        key = k.strip()
        if not key:
            continue
        val: object = v
        if key in {"latency_ms", "digit"} and v.isdigit():
            val = int(v)
        elif key == "confidence":
            val = float(v) if _is_float_str(v) else v
        elif key == "uncertain":
            val = v.lower() in {"1", "true", "yes"}
        out[key] = val
    return out


def _is_float_str(s: str) -> bool:
    if not s:
        return False
    # Accept formats like 0.5, 1, 1.0
    return s.count(".") <= 1 and s.replace(".", "", 1).isdigit()


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
