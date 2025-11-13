from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Final, Literal, Protocol, TypedDict, runtime_checkable

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
        }
        if rid:
            payload["request_id"] = rid
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


class _ConsoleFormatter(logging.Formatter):
    """Pretty, colorized formatter for interactive terminals.

    Highlights:
    - Level tag with color
    - Event/category token emphasized
    - key=value pairs with colored keys and heuristic value coloring
    - Compact UTC timestamp
    """

    _RESET = "\x1b[0m"
    _BOLD = "\x1b[1m"
    _DIM = "\x1b[2m"
    _FG_GRAY = "\x1b[90m"
    _FG_RED = "\x1b[91m"
    _FG_GREEN = "\x1b[92m"
    _FG_YELLOW = "\x1b[93m"
    _FG_BLUE_BRIGHT = "\x1b[94m"
    _FG_MAGENTA = "\x1b[95m"
    _FG_CYAN = "\x1b[36m"
    _FG_WHITE = "\x1b[97m"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        level = record.levelno
        lvl_tag = self._level_tag(level)

        rid = request_id_var.get()
        rid_part = f" {self._DIM}{self._FG_GRAY}rid={rid}{self._RESET}" if rid else ""

        msg = record.getMessage()
        event, kv_pairs, tail = self._split_message(msg)

        parts: list[str] = []
        parts.append(f"{self._DIM}[{ts}]{self._RESET}")
        parts.append(lvl_tag)
        if record.name and record.name != _LOGGER_NAME:
            parts.append(f"{self._DIM}{self._FG_GRAY}{record.name}{self._RESET}")
        if event:
            parts.append(f"{self._BOLD}{self._FG_BLUE_BRIGHT}{event}{self._RESET}")

        for k, v in kv_pairs:
            parts.append(f"{self._DIM}{self._FG_CYAN}{k}{self._RESET}={self._color_value(k, v)}")

        if tail:
            parts.append(tail)

        if record.exc_info:
            exc = self.formatException(record.exc_info)
            parts.append(f"\n{self._FG_RED}{exc}{self._RESET}")

        return " ".join(parts) + rid_part

    def _level_tag(self, level: int) -> str:
        if level >= logging.CRITICAL:
            c = self._FG_MAGENTA
            name = "CRIT"
        elif level >= logging.ERROR:
            c = self._FG_RED
            name = "ERROR"
        elif level >= logging.WARNING:
            c = self._FG_YELLOW
            name = "WARN"
        elif level >= logging.INFO:
            c = self._FG_CYAN
            name = "INFO"
        else:
            c = self._FG_GRAY
            name = "DEBUG"
        return f"{self._BOLD}{c}[{name}]{self._RESET}"

    def _split_message(self, msg: str) -> tuple[str | None, list[tuple[str, str]], str | None]:
        # Structured EVT messages
        if isinstance(msg, str) and msg.startswith("EVT "):
            extra = _parse_evt_fields(msg)
            evt_name = str(extra.pop("event")) if "event" in extra else "event"
            kv_items: list[tuple[str, str]] = [(k, str(v)) for k, v in extra.items()]
            return evt_name, kv_items, None

        if not isinstance(msg, str) or not msg:
            return None, [], None
        toks = msg.split()
        if not toks:
            return None, [], msg

        event: str | None = None
        rest = toks
        if "=" not in toks[0]:
            event = toks[0]
            rest = toks[1:]

        kv: list[tuple[str, str]] = []
        tail_parts: list[str] = []
        for t in rest:
            if "=" in t:
                k, v = t.split("=", 1)
                k = k.strip()
                if k:
                    kv.append((k, v))
                else:
                    tail_parts.append(t)
            else:
                tail_parts.append(t)

        tail = " ".join(tail_parts) if tail_parts else None
        return event, kv, tail

    def _color_value(self, key: str, v: str) -> str:
        ks = key.lower()
        vs = v.strip()
        if ks.endswith("_ms") or ks.endswith("_s") or "time" in ks or ks.endswith("seconds"):
            return f"{self._FG_MAGENTA}{vs}{self._RESET}"
        if "acc" in ks or ks.endswith("_acc"):
            return f"{self._FG_GREEN}{vs}{self._RESET}"
        if vs.lower() in {"true", "false"}:
            return f"{self._FG_CYAN}{vs}{self._RESET}"
        if _is_float_str(vs) or vs.isdigit():
            return f"{self._FG_GREEN}{vs}{self._RESET}"
        return f"{self._FG_WHITE}{vs}{self._RESET}"


class LogEvent(TypedDict, total=False):
    event: str
    latency_ms: int
    digit: int
    confidence: float
    model_id: str
    uncertain: bool


def log_event(event: str, fields: Mapping[str, object] | None = None) -> None:
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


LogStyle = Literal["json", "pretty", "auto"]


def _env_level() -> int:
    v = os.environ.get("HANDWRITING_LOG_LEVEL")
    if not v:
        return logging.INFO
    m = v.strip().upper()
    return {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }.get(m, logging.INFO)


def init_logging(style: LogStyle = "auto") -> logging.Logger:
    """Initialize or refresh the project logger.

    Robust against stdout replacements (e.g., pytest capsys) by re-binding any
    existing StreamHandler to the current sys.stdout and updating its formatter
    and level. Ensures one active StreamHandler without duplicating handlers.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    lvl = _env_level()
    logger.setLevel(lvl)
    # Decide propagation upfront from environment so tests can opt-in
    propagate_env = _env_truthy("HANDWRITING_LOG_PROPAGATE") or _env_truthy("LOG_PROPAGATE")
    logger.propagate = bool(propagate_env)

    # Ensure exactly one StreamHandler bound to current stdout
    formatter = _choose_formatter(style)
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler):
            logger.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(lvl)
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


def _has_stream_handler(logger: logging.Logger) -> bool:
    # Avoid the built-in helper that checks a condition across iterables
    return next(
        (True for h in logger.handlers if isinstance(h, logging.StreamHandler)),
        False,
    )


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    if not v:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


def _choose_formatter(style: LogStyle = "auto") -> logging.Formatter:
    # Explicit style takes precedence
    if style == "json":
        return _JsonFormatter()
    if style == "pretty":
        return _ConsoleFormatter()

    # Auto: honor env and TTY
    force_json = _env_truthy("HANDWRITING_LOG_JSON") or _env_truthy("LOG_JSON")
    force_pretty = _env_truthy("HANDWRITING_LOG_PRETTY") or _env_truthy("LOG_PRETTY")

    @runtime_checkable
    class _HasIsatty(Protocol):
        def isatty(self) -> bool: ...

    out_stream = sys.stdout
    is_tty = isinstance(out_stream, _HasIsatty) and bool(out_stream.isatty())
    if not force_json and (force_pretty or is_tty):
        return _ConsoleFormatter()
    return _JsonFormatter()
