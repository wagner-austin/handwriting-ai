from __future__ import annotations

import logging
import sys

from handwriting_ai.logging import _ConsoleFormatter


def _rec(level: int, msg: object, name: str = "x.other") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="t",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_console_levels_and_other_logger() -> None:
    f = _ConsoleFormatter()
    for lvl in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
        out = f.format(_rec(lvl, "hello", name="other.logger"))
        assert "[" in out and "]" in out and "hello" in out


def test_console_split_message_variants() -> None:
    f = _ConsoleFormatter()
    # non-str message
    out1 = f.format(_rec(logging.INFO, 123))
    assert "[INFO]" in out1
    # empty message
    out0 = f.format(_rec(logging.INFO, ""))
    assert "[INFO]" in out0
    # whitespace-only message
    out2 = f.format(_rec(logging.INFO, "   "))
    assert "[INFO]" in out2
    # key with empty name should be treated as tail
    out3 = f.format(_rec(logging.INFO, "foo =2 bar=3"))
    assert "foo" in out3 and "=2" in out3 and "bar" in out3
    # booleans and numbers get colored values
    out4 = f.format(_rec(logging.INFO, "flag=true n=5 time_s=1.2"))
    assert "flag" in out4 and "true" in out4 and "n" in out4 and "5" in out4 and "time_s" in out4


def test_console_with_exc_info() -> None:
    f = _ConsoleFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        # Touch logger to satisfy guard for non-silent except in tests
        import logging as _logging

        _logging.getLogger("handwriting_ai").info("exc_captured_for_test")
        exc = sys.exc_info()
    rec = logging.LogRecord(
        name="x.other",
        level=logging.ERROR,
        pathname="t",
        lineno=1,
        msg="evt something happened",
        args=(),
        exc_info=exc,
    )
    out = f.format(rec)
    assert "boom" in out and "Traceback" in out
