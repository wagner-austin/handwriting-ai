from __future__ import annotations

import logging

import pytest
from scripts.rq_keyspace_watcher import main as _main


def test_rq_keyspace_watcher_ok(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    called = {"n": 0}

    def _run() -> None:
        called["n"] += 1

    monkeypatch.setattr("scripts.rq_keyspace_watcher.run_notify_from_env", _run, raising=True)
    caplog.set_level(logging.INFO, logger="handwriting_ai")
    rc = _main()
    assert rc == 0 and called["n"] == 1


def test_rq_keyspace_watcher_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.rq_keyspace_watcher.run_notify_from_env", _boom, raising=True)
    assert _main() == 1
