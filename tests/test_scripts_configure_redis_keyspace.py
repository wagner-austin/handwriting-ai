from __future__ import annotations

import sys
from types import ModuleType

import pytest
from scripts.configure_redis_keyspace import configure_keyspace_notifications


class _Client:
    def __init__(self, configured: bool, fail_set: bool = False) -> None:
        self._configured = configured
        self._fail_set = fail_set
        self.closed = False

    def ping(self) -> bool:
        return True

    def config_get(self, key: str) -> dict[str, str]:
        return {"notify-keyspace-events": "Kz" if self._configured else "K"}

    def config_set(self, key: str, value: str) -> None:
        if self._fail_set:
            raise RuntimeError("set failed")
        self._configured = True

    def close(self) -> None:
        self.closed = True


class _Redis(ModuleType):
    def __init__(self, name: str, configured: bool, fail_set: bool = False) -> None:
        super().__init__(name)
        self._configured = configured
        self._fail_set = fail_set
        # Expose a Redis attribute that forwards to this instance
        self.Redis = self

    def from_url(self, url: str, decode_responses: bool) -> _Client:
        assert decode_responses is True
        return _Client(self._configured, self._fail_set)


def test_configure_already_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "redis", _Redis("redis", configured=True))
    configure_keyspace_notifications("redis://x")


def test_configure_sets_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "redis", _Redis("redis", configured=False))
    configure_keyspace_notifications("redis://x")


def test_configure_raises_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RedisErr(_Redis):
        class RedisError(Exception):
            pass

    monkeypatch.setitem(sys.modules, "redis", _RedisErr("redis", configured=False, fail_set=True))
    with pytest.raises(RuntimeError):
        configure_keyspace_notifications("redis://x")


def test_configure_ping_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ClientPingBad(_Client):
        def ping(self) -> bool:
            return False

    class _RedisPingBad(_Redis):
        class RedisError(Exception):
            pass

        def from_url(self, url: str, decode_responses: bool) -> _Client:
            return _ClientPingBad(False)

    monkeypatch.setitem(sys.modules, "redis", _RedisPingBad("redis", configured=False))
    with pytest.raises(RuntimeError):
        configure_keyspace_notifications("redis://x")


def test_configure_verify_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ClientVerifyBad(_Client):
        def __init__(self) -> None:
            super().__init__(configured=False)

        def config_get(self, key: str) -> dict[str, str]:
            return {"notify-keyspace-events": "K"}

    class _RedisVerifyBad(_Redis):
        class RedisError(Exception):
            pass

        def from_url(self, url: str, decode_responses: bool) -> _Client:
            return _ClientVerifyBad()

    monkeypatch.setitem(sys.modules, "redis", _RedisVerifyBad("redis", configured=False))
    with pytest.raises(RuntimeError):
        configure_keyspace_notifications("redis://x")


def test_main_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    monkeypatch.delenv("REDIS_URL", raising=False)
    mod = importlib.reload(importlib.import_module("scripts.configure_redis_keyspace"))
    with pytest.raises(SystemExit):
        mod.main()


def test_main_success(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import importlib
    import sys

    def _ok(url: str) -> None:
        return None

    monkeypatch.setenv("REDIS_URL", "redis://x")
    sys.modules.pop("scripts.configure_redis_keyspace", None)
    mod = importlib.import_module("scripts.configure_redis_keyspace")
    monkeypatch.setattr(mod, "configure_keyspace_notifications", _ok, raising=True)
    # Call main() directly - this is what __main__ does, so we get the same coverage
    mod.main()
    out = capsys.readouterr().out
    assert "Configuration complete" in out


def test_main_error_logs_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    def _boom(url: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setenv("REDIS_URL", "redis://x")
    mod = importlib.reload(importlib.import_module("scripts.configure_redis_keyspace"))
    monkeypatch.setattr(mod, "configure_keyspace_notifications", _boom, raising=True)
    with pytest.raises(RuntimeError):
        mod.main()
