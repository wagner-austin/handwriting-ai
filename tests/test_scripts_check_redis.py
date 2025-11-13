from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Protocol

import pytest


class _RedisClientProto(Protocol):
    def ping(self) -> bool: ...
    def config_get(self, key: str) -> dict[str, str]: ...
    def dbsize(self) -> int: ...
    def keys(self, pattern: str) -> list[str]: ...
    def type(self, key: str) -> str: ...
    def zcard(self, key: str) -> int: ...


def test_check_redis_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    import importlib

    mod = importlib.import_module("scripts.check_redis")
    with pytest.raises(SystemExit) as ei:
        mod.main()
    code = ei.value.code
    assert (isinstance(code, int) and code == 1) or (isinstance(code, str) and code == "1")


def test_check_redis_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _StubClient:
        def ping(self) -> bool:
            return True

        def config_get(self, key: str) -> dict[str, str]:
            return {"notify-keyspace-events": "Kz"}

        def dbsize(self) -> int:
            return 3

        def keys(self, pattern: str) -> list[str]:
            if pattern.startswith("rq:queue:"):
                return ["rq:queue:digits"]
            if pattern.startswith("rq:registry:"):
                return ["rq:registry:failed:digits"]
            if pattern == "*":
                return ["a", "b", "rq:queue:digits"]
            return []

        def type(self, key: str) -> str:
            return "zset" if key.startswith("rq:registry:") else "string"

        def zcard(self, key: str) -> int:
            return 0

    class _StubRedis(ModuleType):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            # Point Redis attribute to this instance to mimic redis.Redis API
            self.Redis = self

        def from_url(
            self,
            url: str,
            decode_responses: bool,
            socket_timeout: int,
            socket_connect_timeout: int,
        ) -> _RedisClientProto:
            assert decode_responses is True
            assert socket_timeout > 0 and socket_connect_timeout > 0
            return _StubClient()

    monkeypatch.setitem(sys.modules, "redis", _StubRedis("redis"))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    # Ensure structured logs propagate to root for caplog
    monkeypatch.setenv("HANDWRITING_LOG_PROPAGATE", "1")

    import importlib

    mod = importlib.reload(importlib.import_module("scripts.check_redis"))
    caplog.set_level(logging.INFO, logger="handwriting_ai")
    mod.main()
    messages = [rec.message for rec in caplog.records if rec.name == "handwriting_ai"]
    assert any("redis_check_rq_queues" in m for m in messages)
    assert any("redis_check_total_keys" in m for m in messages)


def test_check_redis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubClient:
        def ping(self) -> bool:
            # Fail inside try-block so the script's handler runs
            raise RuntimeError("connect-failed")

        def config_get(self, key: str) -> dict[str, str]:
            return {}

        def dbsize(self) -> int:
            return 0

        def keys(self, pattern: str) -> list[str]:
            return []

        def type(self, key: str) -> str:
            return "string"

        def zcard(self, key: str) -> int:
            return 0

    class _Boom(ModuleType):
        class Exceptions:
            class RedisError(Exception):
                pass

        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.Redis = self
            self.exceptions = self.Exceptions

        def from_url(
            self,
            url: str,
            decode_responses: bool,
            socket_timeout: int,
            socket_connect_timeout: int,
        ) -> _StubClient:
            return _StubClient()

    monkeypatch.setitem(sys.modules, "redis", _Boom("redis"))
    monkeypatch.setenv("REDIS_URL", "redis://bad")

    import importlib

    mod = importlib.reload(importlib.import_module("scripts.check_redis"))
    with pytest.raises(SystemExit):
        mod.main()


def test_check_redis_main_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    monkeypatch.delenv("REDIS_URL", raising=False)
    sys.modules.pop("scripts.check_redis", None)
    with pytest.raises(SystemExit):
        runpy.run_module("scripts.check_redis", run_name="__main__")
