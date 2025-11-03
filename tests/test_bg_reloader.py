from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


def _count_reloader_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name == "model-reloader")


def test_background_reloader_starts_and_stops_cleanly() -> None:
    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    app = create_app(s, reload_interval_seconds=0.05)
    with TestClient(app):
        # Thread should be running shortly after startup
        for _ in range(20):
            if _count_reloader_threads() > 0:
                break
            time.sleep(0.01)
        assert _count_reloader_threads() > 0
    # On shutdown, thread should stop promptly
    for _ in range(50):
        if _count_reloader_threads() == 0:
            break
        time.sleep(0.02)
    assert _count_reloader_threads() == 0
