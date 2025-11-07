from __future__ import annotations

import sys
from pathlib import Path

import pytest
import scripts.rq_worker as rw

import handwriting_ai.jobs.digits as dj
from handwriting_ai.training.mnist_train import TrainConfig


def test_rq_worker_patches_training_and_calls_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    # Stub Redis connection
    class _Conn:
        def ping(self) -> bool:
            return True

    def _rf(url: str) -> _Conn:
        _ = url
        return _Conn()

    monkeypatch.setattr(rw, "_redis_from_url", _rf, raising=True)

    # Stub rq module expected by scripts.rq_worker
    class _Queue:
        def __init__(self, name: str, connection: object | None = None) -> None:
            created["queue_name"] = name

    class _Worker:
        def __init__(self, queues: list[_Queue], **kwargs: object) -> None:
            created["worker_init"] = True
            # Ensure handler wiring is passed
            handlers = kwargs.get("exception_handlers")
            created["has_handlers"] = bool(handlers)

        def work(self) -> bool:  # pragma: no cover - simple boolean path
            created["worked"] = True
            return True

    class _RQStub:
        def __init__(self) -> None:
            self.Queue: type[_Queue] = _Queue
            self.Worker: type[_Worker] = _Worker

        def push_connection(self, conn: object) -> None:  # pragma: no cover - trivial
            _ = conn
            created["pushed"] = True

        def pop_connection(self) -> None:  # pragma: no cover - trivial
            created["popped"] = True

    monkeypatch.setitem(sys.modules, "rq", _RQStub())

    # Sentinel training function to verify patch happened
    def _sentinel(cfg: TrainConfig) -> Path:
        _ = cfg
        created["patched"] = True
        return Path(".")

    monkeypatch.setattr(rw, "_real_run_training", _sentinel, raising=True)

    # Precondition: ensure digits job still points to default before start
    assert callable(dj._run_training)

    # Call the worker start with fake settings
    code = rw._start_worker(rw._RqSettings(redis_url="redis://fake", queue_name="digits"))
    assert code == 0

    # Verify wiring and rq interactions
    assert created.get("queue_name") == "digits"
    assert created.get("worker_init") and created.get("worked")
    assert created.get("pushed") and created.get("popped")
    # Training function should have been patched to our sentinel
    assert dj._run_training is _sentinel
