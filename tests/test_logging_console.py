from __future__ import annotations

import logging
import os

from handwriting_ai.logging import (
    _choose_formatter,
    _ConsoleFormatter,
)


def test_console_formatter_evt_line() -> None:
    msg = (
        "EVT event=train_batch_done epoch=1/1 batch=111/118 "
        "batch_loss=0.0396 batch_acc=0.9883 avg_loss=0.1998 samples_per_sec=163.0"
    )
    rec = logging.LogRecord(
        name="handwriting_ai",
        level=logging.INFO,
        pathname="test",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    f = _ConsoleFormatter()
    out = f.format(rec)
    # Contains INFO tag and key details (keys/values may be colorized)
    assert "[INFO]" in out
    assert "train_batch_done" in out
    assert "epoch" in out and "1/1" in out
    assert "batch_acc" in out and "0.9883" in out


def test_choose_formatter_explicit() -> None:
    # json should produce a JSON object string
    rec = logging.LogRecord(
        name="handwriting_ai",
        level=logging.INFO,
        pathname="t",
        lineno=1,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    f_json = _choose_formatter("json")
    out_json = f_json.format(rec)
    assert out_json.strip().startswith("{") and '"message"' in out_json
    # pretty should include ANSI escapes and readable content
    f_pretty = _choose_formatter("pretty")
    out_pretty = f_pretty.format(rec)
    assert "\x1b[" in out_pretty and "hello" in out_pretty


def test_choose_formatter_env_overrides() -> None:
    # Force pretty via env, then force json
    old_pretty = os.environ.get("HANDWRITING_LOG_PRETTY")
    old_json = os.environ.get("HANDWRITING_LOG_JSON")
    try:
        os.environ["HANDWRITING_LOG_PRETTY"] = "1"
        f1 = _choose_formatter("auto")
        rec_env = logging.LogRecord(
            name="handwriting_ai",
            level=logging.INFO,
            pathname="t",
            lineno=1,
            msg="env pretty",
            args=(),
            exc_info=None,
        )
        out1 = f1.format(rec_env)
        assert "\x1b[" in out1 and "pretty" in out1
        os.environ.pop("HANDWRITING_LOG_PRETTY", None)
        os.environ["HANDWRITING_LOG_JSON"] = "yes"
        f2 = _choose_formatter("auto")
        rec = logging.LogRecord(
            name="handwriting_ai",
            level=logging.INFO,
            pathname="t",
            lineno=1,
            msg="a=b",
            args=(),
            exc_info=None,
        )
        s = f2.format(rec)
        assert s.strip().startswith("{") and '"message"' in s
    finally:
        if old_pretty is None:
            os.environ.pop("HANDWRITING_LOG_PRETTY", None)
        else:
            os.environ["HANDWRITING_LOG_PRETTY"] = old_pretty
        if old_json is None:
            os.environ.pop("HANDWRITING_LOG_JSON", None)
        else:
            os.environ["HANDWRITING_LOG_JSON"] = old_json
