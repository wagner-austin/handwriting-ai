from __future__ import annotations

import logging
from io import StringIO

import pytest

import handwriting_ai.training.memory_diagnostics as md
from handwriting_ai.logging import _JsonFormatter, get_logger
from handwriting_ai.monitoring import (
    CgroupMemoryBreakdown,
    CgroupMemoryUsage,
    MemorySnapshot,
    ProcessMemory,
)


def _mk_snap(usage_mb: int, limit_mb: int) -> MemorySnapshot:
    return MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=0),
        workers=(),
        cgroup_usage=CgroupMemoryUsage(
            usage_bytes=usage_mb * 1024 * 1024,
            limit_bytes=limit_mb * 1024 * 1024,
            percent=(float(usage_mb) / float(limit_mb)) * 100.0,
        ),
        cgroup_breakdown=CgroupMemoryBreakdown(
            anon_bytes=0, file_bytes=0, kernel_bytes=0, slab_bytes=0
        ),
    )


def test_empty_history_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    md.reset_diagnostics()

    # Provide a current snapshot so get_memory_snapshot() is not called implicitly
    snap = _mk_snap(usage_mb=100, limit_mb=1000)
    diag = md.get_memory_diagnostics(snapshot=snap)

    assert diag.current_mb == 100
    assert diag.peak_mb == 100
    assert diag.avg_mb == 100
    assert diag.growth_rate_mb_per_batch == 0.0
    assert diag.volatility_mb == 0.0
    assert diag.trend == "stable"
    assert diag.window_size == 0


def test_record_and_linear_growth() -> None:
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=10)

    limit_mb = 1000
    # Linear growth: 100, 110, 120, 130 MB across 4 batches
    for mb in (100, 110, 120, 130):
        md.record_batch_memory(snapshot=_mk_snap(usage_mb=mb, limit_mb=limit_mb))

    diag = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=130, limit_mb=limit_mb))
    assert diag.current_mb == 130
    assert diag.peak_mb == 130
    assert diag.avg_mb == 115
    # Slope should be close to 10 MB/batch
    assert abs(diag.growth_rate_mb_per_batch - 10.0) < 1e-6
    assert diag.volatility_mb == 0.0
    assert diag.trend in {"stable", "growing"}  # linear growth should not be critical


def test_volatility_and_trend_thresholds() -> None:
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=10)
    limit_mb = 2000
    # Fluctuating usage to drive volatility: 100, 160, 120, 200 (deltas: +60, -40, +80)
    for mb in (100, 160, 120, 200):
        md.record_batch_memory(snapshot=_mk_snap(usage_mb=mb, limit_mb=limit_mb))

    diag = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=200, limit_mb=limit_mb))
    assert diag.volatility_mb > 10.0
    assert diag.trend in {"growing", "critical"}

    # Drive critical by big jumps
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=10)
    for mb in (100, 300, 500):
        md.record_batch_memory(snapshot=_mk_snap(usage_mb=mb, limit_mb=limit_mb))
    diag2 = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=500, limit_mb=limit_mb))
    assert diag2.trend in {"growing", "critical"}


def test_trend_critical_by_growth() -> None:
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=10)
    limit_mb = 5000
    # Large steady jumps to exceed critical growth threshold
    for mb in (100, 400, 700, 1000):
        md.record_batch_memory(snapshot=_mk_snap(usage_mb=mb, limit_mb=limit_mb))
    diag = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=1000, limit_mb=limit_mb))
    assert diag.trend in {"critical", "growing"} and diag.growth_rate_mb_per_batch > 20.0


def test_growth_rate_denominator_zero_internal() -> None:
    # Directly exercise internal guard for zero denominator
    from collections import deque as _dq

    snaps = _dq([(1, 100), (1, 200)], maxlen=10)
    assert md._compute_growth_rate(snaps) == 0.0


def test_history_window_minimum_size() -> None:
    # window_size <= 0 coerces to 1
    h = md.MemoryHistory(window_size=0)
    # Record ensures object is used; behavior should not error
    h.record_batch(snapshot=_mk_snap(usage_mb=100, limit_mb=1000))
    diag = h.get_diagnostics(limit_mb=1000, current_snapshot=_mk_snap(usage_mb=100, limit_mb=1000))
    assert diag.window_size == 1


def test_record_batch_memory_auto_init() -> None:
    # Ensure record_batch_memory initializes history when None
    md.reset_diagnostics()
    md.record_batch_memory(snapshot=_mk_snap(usage_mb=111, limit_mb=1000))
    d = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=111, limit_mb=1000))
    assert d.window_size >= 1


def test_reset_diagnostics_no_history_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise the branch where _history is None during reset
    import handwriting_ai.training.memory_diagnostics as _md

    monkeypatch.setattr(_md, "_history", None, raising=False)
    _md.reset_diagnostics()
    # Ensure still usable after
    _md.record_batch_memory(snapshot=_mk_snap(usage_mb=50, limit_mb=1000))
    d = _md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=50, limit_mb=1000))
    assert d.window_size >= 1


def test_predict_oom_branches() -> None:
    # Non-positive growth -> None
    assert md._predict_oom(100, 1000, 0.0) is None
    assert md._predict_oom(100, 1000, -1.0) is None
    # Already OOM: remaining <= 0 -> 0
    assert md._predict_oom(1000, 1000, 1.0) == 0
    assert md._predict_oom(1100, 1000, 10.0) == 0
    # Large prediction -> None
    assert md._predict_oom(100, 2000, 0.5) is None  # (1900 / 0.5) = 3800
    # Small prediction -> int
    assert md._predict_oom(900, 1000, 10.0) == 10


def test_reset_clears_history() -> None:
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=5)
    md.record_batch_memory(snapshot=_mk_snap(usage_mb=100, limit_mb=1000))
    diag1 = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=100, limit_mb=1000))
    assert diag1.window_size == 1
    md.reset_diagnostics()
    diag2 = md.get_memory_diagnostics(snapshot=_mk_snap(usage_mb=100, limit_mb=1000))
    assert diag2.window_size == 0


def test_log_memory_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure log output contains diagnostics fields and honors provided snapshot."""
    md.reset_diagnostics()
    md.initialize_diagnostics(window_size=5)
    snap = _mk_snap(usage_mb=123, limit_mb=1000)
    md.record_batch_memory(snapshot=snap)

    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        # Use the same snapshot to avoid internal re-sampling
        md.log_memory_diagnostics(context="test_ctx", snapshot=snap)
    finally:
        logger.removeHandler(handler)

    out = buf.getvalue()
    assert "test_ctx memory_diagnostics" in out
    assert "current_mb=123" in out
    assert "window_size=1" in out
