from __future__ import annotations

import sys
from types import ModuleType

import pytest

import handwriting_ai.monitoring as mon


def test_cgroup_memory_limit_import_error_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Replace resources module with one missing the expected symbol
    key = "handwriting_ai.training.resources"
    original: ModuleType | None = sys.modules.get(key)
    try:
        dummy = ModuleType(key)
        sys.modules[key] = dummy
        out = mon._cgroup_memory_limit_bytes()
        assert out is None
    finally:
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


def test_compute_pressure_percent_with_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force a concrete limit path and verify ratio calculation and clamping
    monkeypatch.setattr(mon, "_cgroup_memory_limit_bytes", lambda: 1_000_000_000, raising=True)
    pct = mon._compute_pressure_percent(rss_bytes=500_000_000, fallback_vm_percent=12.0)
    assert abs(pct - 50.0) < 1e-6

    pct2 = mon._compute_pressure_percent(rss_bytes=50_000_000_000, fallback_vm_percent=12.0)
    assert pct2 == 1000.0  # clamped upper bound
