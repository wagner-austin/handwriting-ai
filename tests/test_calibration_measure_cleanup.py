from __future__ import annotations

import types
from collections.abc import Generator

import pytest
import torch as _t

import handwriting_ai.training.calibration.measure as meas


def _one_batch_loader() -> Generator[tuple[_t.Tensor, _t.Tensor], None, None]:
    x = _t.zeros((1, 1, 28, 28), dtype=_t.float32)
    y = _t.zeros((1,), dtype=_t.long)
    yield x, y


def test_measure_training_calls_gc_collect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _collect() -> int:  # return type matches gc.collect signature
        calls["n"] += 1
        return 0

    monkeypatch.setattr(meas, "_gc", types.SimpleNamespace(collect=_collect), raising=True)

    sps, p95, peak_pct, exceeded = meas._measure_training(
        ds_len=1,
        loader=_one_batch_loader(),
        k=1,
        device=_t.device("cpu"),
        batch_size_hint=1,
    )
    assert sps >= 0.0 and p95 >= 0.0 and peak_pct >= 0.0 and exceeded in {True, False}
    assert calls["n"] >= 1
