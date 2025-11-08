from __future__ import annotations

import json
from pathlib import Path

import pytest

import handwriting_ai.jobs.digits as dj
from handwriting_ai.training.mnist_train import TrainConfig


class _Pub:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def publish(self, channel: str, message: str) -> int:
        self.sent.append((channel, message))
        return 1


def _payload() -> dict[str, object]:
    return {
        "type": "digits.train.v1",
        "request_id": "r-1",
        "user_id": 42,
        "model_id": "m-1",
        "epochs": 1,
        "batch_size": 1,
        "lr": 0.001,
        "seed": 7,
        "augment": False,
        "notes": None,
    }


def test_failed_message_includes_exception_text(monkeypatch: pytest.MonkeyPatch) -> None:
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)

    def _raise_run(_: TrainConfig) -> Path:
        raise RuntimeError("unique-boom")

    monkeypatch.setattr(dj, "_run_training", _raise_run)
    with pytest.raises(RuntimeError):
        dj.process_train_job(_payload())

    # Parse last failed event
    msgs = [m for _, m in p.sent]
    failed = [m for m in msgs if "failed" in m][-1]
    obj: dict[str, object] = json.loads(failed)
    assert obj["type"] == "digits.train.failed.v1"
    m1: object = obj["message"]
    assert isinstance(m1, str)
    assert "RuntimeError" in m1 and "unique-boom" in m1


def test_failed_message_for_memory_guard_includes_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = _Pub()
    monkeypatch.setenv("DIGITS_EVENTS_CHANNEL", "digits:events")
    monkeypatch.setattr(dj, "_make_publisher", lambda: p)

    def _raise_run(_: TrainConfig) -> Path:
        raise RuntimeError("memory_pressure_guard_triggered")

    # Ensure memory guard threshold is controlled
    import handwriting_ai.training.safety as safety

    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=88.0, required_consecutive=3)
    )

    monkeypatch.setattr(dj, "_run_training", _raise_run)
    with pytest.raises(RuntimeError):
        dj.process_train_job(_payload())

    msgs = [m for _, m in p.sent]
    failed = [m for m in msgs if "failed" in m][-1]
    obj: dict[str, object] = json.loads(failed)
    assert obj["type"] == "digits.train.failed.v1"
    m2: object = obj["message"]
    assert isinstance(m2, str)
    assert "memory pressure" in m2.lower()
    assert "88.0" in m2
