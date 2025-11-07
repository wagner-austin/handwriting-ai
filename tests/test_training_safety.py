from __future__ import annotations

import pytest

import handwriting_ai.training.safety as safety


def test_memory_guard_consecutive_logic(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure guard to trigger after 2 consecutive True checks
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=90.0, required_consecutive=2)
    )
    safety.reset_memory_guard()

    # Patch check_memory_pressure path in safety via module attribute
    # by swapping dependency in the module closure
    def _always(*, threshold_percent: float) -> bool:
        return True

    monkeypatch.setattr(safety, "check_memory_pressure", _always, raising=True)
    assert safety.on_batch_check() is False  # first True
    assert safety.on_batch_check() is True  # second True triggers


def test_memory_guard_resets_on_relief(monkeypatch: pytest.MonkeyPatch) -> None:
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=90.0, required_consecutive=2)
    )
    safety.reset_memory_guard()
    # Alternate True/False so it never triggers
    seq = iter([True, False, True, False])

    def _seq(*, threshold_percent: float) -> bool:
        return next(seq, False)


def test_memory_guard_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=False, threshold_percent=95.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Even with True pressure, disabled guard returns False
    def _always2(*, threshold_percent: float) -> bool:
        return True

    monkeypatch.setattr(safety, "check_memory_pressure", _always2, raising=True)
    assert safety.on_batch_check() is False
