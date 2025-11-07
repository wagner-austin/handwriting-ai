from __future__ import annotations

import sys

import pytest
import scripts.rq_worker as rw


def test_worker_receives_exception_handlers_kw(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class _Conn:
        def ping(self) -> bool:
            return True

    def _rf(url: str) -> _Conn:
        _ = url
        return _Conn()

    monkeypatch.setattr(rw, "_redis_from_url", _rf, raising=True)

    class _Queue:
        def __init__(self, name: str, connection: object | None = None) -> None:
            created["queue_name"] = name

    class _Worker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            created["exception_handlers"] = kwargs.get("exception_handlers")

        def work(self) -> bool:  # pragma: no cover - trivial
            return True

    class _RQ:
        Queue: type[_Queue]
        Worker: type[_Worker]

        def __init__(self) -> None:
            self.Queue = _Queue
            self.Worker = _Worker

        def push_connection(self, conn: object) -> None:  # pragma: no cover - trivial
            _ = conn

        def pop_connection(self) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setitem(sys.modules, "rq", _RQ())

    code = rw._start_worker(rw._RqSettings(redis_url="redis://fake", queue_name="digits"))
    assert code == 0
    handlers = created.get("exception_handlers")
    assert isinstance(handlers, list) and handlers and callable(handlers[0])


def test_worker_fallback_push_exc_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {"pushed": False}

    class _Conn:
        def ping(self) -> bool:
            return True

    def _rf(url: str) -> _Conn:
        _ = url
        return _Conn()

    monkeypatch.setattr(rw, "_redis_from_url", _rf, raising=True)

    class _Queue:
        def __init__(self, name: str, connection: object | None = None) -> None:
            _ = name

    # No fallback path: constructing Worker must accept exception_handlers kw
    class _Worker2:
        def __init__(self, *args: object, **kwargs: object) -> None:
            # Ensure we received exception_handlers
            handlers = kwargs.get("exception_handlers")
            created["handlers"] = handlers

        def work(self) -> bool:  # pragma: no cover - trivial
            return True

    class _RQ2:
        Queue: type[_Queue]
        Worker: type[_Worker2]

        def __init__(self) -> None:
            self.Queue = _Queue
            self.Worker = _Worker2

        def push_connection(self, conn: object) -> None:  # pragma: no cover - trivial
            _ = conn

        def pop_connection(self) -> None:  # pragma: no cover - trivial
            pass

    monkeypatch.setitem(sys.modules, "rq", _RQ2())

    code = rw._start_worker(rw._RqSettings(redis_url="redis://fake", queue_name="digits"))
    assert code == 0
    assert isinstance(created.get("handlers"), list)
