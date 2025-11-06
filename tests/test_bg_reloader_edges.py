from __future__ import annotations

import threading

import pytest

from handwriting_ai.api.app import (
    _debug_invoke_reloader_start,
    _debug_invoke_reloader_stop,
    create_app,
)
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def test_reloader_stop_without_startup() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s, reload_interval_seconds=0.05)
    # Invoke stop before any start to hit both-None path
    _debug_invoke_reloader_stop(app)


def test_reloader_debug_noop_when_disabled() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    # Disabled interval means no handlers registered; debug helpers should no-op
    app = create_app(s, reload_interval_seconds=0.0)
    _debug_invoke_reloader_start(app)
    _debug_invoke_reloader_stop(app)


def test_reloader_stop_with_event_but_no_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s, reload_interval_seconds=0.05)

    class _Boom:
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("no thread")

    # Force thread creation to fail after event is created in the start handler
    monkeypatch.setattr(threading, "Thread", _Boom, raising=True)
    with pytest.raises(RuntimeError):
        _debug_invoke_reloader_start(app)
    # Now stop should see stop_evt set while thread is None
    _debug_invoke_reloader_stop(app)
