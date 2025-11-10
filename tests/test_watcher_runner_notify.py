from __future__ import annotations

import pytest

import handwriting_ai.jobs.watcher.runner as runner


def test_run_notify_from_env_parses_queues_and_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, tuple[str, tuple[str, ...], str] | None] = {"args": None}

    class _FakeWatcher:
        def __init__(self, redis_url: str, *, queues: tuple[str, ...], events_channel: str) -> None:
            called["args"] = (redis_url, queues, events_channel)

        def run_forever(self) -> None:  # pragma: no cover - no-op
            return None

    monkeypatch.setenv("REDIS_URL", "redis://localhost/1")
    monkeypatch.setenv("RQ__QUEUES", "a,b")
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(runner, "NotificationWatcher", _FakeWatcher, raising=True)

    runner.run_notify_from_env()
    assert called["args"] == ("redis://localhost/1", ("a", "b"), "digits:events")


def test_run_notify_from_env_requires_redis_url(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("REDIS_URL", "RQ__QUEUES", "DIGITS_EVENTS_CHANNEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        runner.run_notify_from_env()


def test_run_notify_from_env_wildcard(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, tuple[str, tuple[str, ...], str] | None] = {"args": None}

    class _FakeWatcher:
        def __init__(self, redis_url: str, *, queues: tuple[str, ...], events_channel: str) -> None:
            called["args"] = (redis_url, queues, events_channel)

        def run_forever(self) -> None:  # pragma: no cover - no-op
            return None

    monkeypatch.setenv("REDIS_URL", "redis://localhost/1")
    monkeypatch.setenv("RQ__QUEUES", "*")
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(runner, "NotificationWatcher", _FakeWatcher, raising=True)

    runner.run_notify_from_env()
    assert called["args"] == ("redis://localhost/1", ("*",), "digits:events")
