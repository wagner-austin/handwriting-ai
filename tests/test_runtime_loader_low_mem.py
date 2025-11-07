from __future__ import annotations

import pytest

import handwriting_ai.training.runtime as rt
from handwriting_ai.training.resources import ResourceLimits


class _Cfg:
    def __init__(self, threads: int, batch_size: int) -> None:
        self._threads = threads
        self._bs = batch_size

    @property
    def threads(self) -> int:
        return self._threads

    @property
    def batch_size(self) -> int:
        return self._bs


def test_loader_prefetch_reduced_under_low_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1 * 1024 * 1024 * 1024,
        optimal_threads=2,
        optimal_workers=0,
        max_batch_size=64,
    )
    monkeypatch.setattr(rt, "detect_resource_limits", lambda: limits, raising=True)
    cfg = _Cfg(threads=0, batch_size=64)
    ec, _ = rt.build_effective_config(cfg)
    assert ec.loader_cfg.prefetch_factor == 1
    assert ec.loader_cfg.num_workers == 0
    assert ec.loader_cfg.persistent_workers is False
