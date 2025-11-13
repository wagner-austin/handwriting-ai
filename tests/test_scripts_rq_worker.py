from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Protocol

import pytest
import scripts.rq_worker as rw


def test_get_env_str_and_load_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://x")
    monkeypatch.setenv("RQ__QUEUE", "digits")
    s = rw._load_rq_settings()
    assert s.redis_url.endswith("x") and s.queue_name == "digits"


class _Conn(Protocol):
    def ping(self) -> bool: ...

    decode_responses: bool


class _StubConn:
    def __init__(self, ok: bool, decode_responses: bool = False) -> None:
        self._ok = ok
        self.decode_responses = decode_responses

    def ping(self) -> bool:
        return self._ok


def test_start_worker_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Q:
        def __init__(self, name: str, connection: _Conn) -> None:
            self._name = name
            self._conn = connection

    class _W:
        def __init__(self, queues: list[_Q], connection: _Conn) -> None:
            self._qs = queues
            self._conn = connection

        def work(self) -> bool:
            return True

    class _RQMod(ModuleType):
        Queue = _Q
        Worker = _W

        @staticmethod
        def push_connection(conn: _Conn) -> None:
            return None

        @staticmethod
        def pop_connection() -> None:
            return None

    # Patch redis connection factory and rq module
    def _conn_factory(url: str) -> _StubConn:
        return _StubConn(True, False)

    monkeypatch.setattr(rw, "_redis_from_url", _conn_factory, raising=True)
    monkeypatch.setitem(os.environ, "REDIS_URL", "redis://x")
    monkeypatch.setitem(os.environ, "RQ__QUEUE", "digits")
    monkeypatch.setitem(sys.modules, "rq", _RQMod("rq"))
    rc = rw._start_worker(rw._load_rq_settings())
    assert rc == 0


def test_start_worker_ping_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _conn_factory_bad(url: str) -> _StubConn:
        return _StubConn(False)

    monkeypatch.setattr(rw, "_redis_from_url", _conn_factory_bad, raising=True)
    with pytest.raises(RuntimeError):
        rw._start_worker(rw._RqSettings(redis_url="redis://x", queue_name="digits"))


def test_start_worker_ping_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ConnBoom(_StubConn):
        def ping(self) -> bool:
            raise RuntimeError("boom")

    def _factory(url: str) -> _ConnBoom:
        return _ConnBoom(True)

    monkeypatch.setattr(rw, "_redis_from_url", _factory, raising=True)
    with pytest.raises(RuntimeError):
        rw._start_worker(rw._RqSettings(redis_url="redis://x", queue_name="digits"))


def test_start_worker_no_push_pop(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Q:
        def __init__(self, name: str, connection: _Conn) -> None:
            self._name = name
            self._conn = connection

    class _W:
        def __init__(self, queues: list[_Q], connection: _Conn) -> None:
            self._qs = queues
            self._conn = connection

        def work(self) -> bool:
            return True

    class _RQMini(ModuleType):
        Queue = _Q
        Worker = _W

    def _conn_factory(url: str) -> _StubConn:
        return _StubConn(True, False)

    monkeypatch.setattr(rw, "_redis_from_url", _conn_factory, raising=True)
    monkeypatch.setitem(os.environ, "REDIS_URL", "redis://x")
    monkeypatch.setitem(os.environ, "RQ__QUEUE", "digits")
    monkeypatch.setitem(sys.modules, "rq", _RQMini("rq"))
    rc = rw._start_worker(rw._load_rq_settings())
    assert rc == 0


def test_main_returns_1_on_worker_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    # main() should return 1 when _start_worker raises
    monkeypatch.setenv("REDIS_URL", "redis://x")
    monkeypatch.setenv("RQ__QUEUE", "digits")

    def _raise(_: object) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(rw, "_start_worker", _raise, raising=True)
    assert rw.main([]) == 1
