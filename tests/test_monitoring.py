from __future__ import annotations

import logging
from io import StringIO
from types import SimpleNamespace

import pytest

import handwriting_ai.monitoring as mon
from handwriting_ai.logging import _JsonFormatter, get_logger


class _DummyVM:
    def __init__(self, total: int, available: int, percent: float) -> None:
        self.total = total
        self.available = available
        self.percent = percent


class _DummyProc:
    def __init__(self, rss: int, percent: float) -> None:
        self._rss = rss
        self._percent = percent

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self._rss)

    def memory_percent(self) -> float:
        return self._percent


def test_get_memory_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch psutil within the module namespace using a stub object
    dummy = SimpleNamespace(
        Process=lambda: _DummyProc(rss=150 * 1024 * 1024, percent=12.5),
        virtual_memory=lambda: _DummyVM(
            total=4 * 1024 * 1024 * 1024,
            available=3 * 1024 * 1024 * 1024,
            percent=25.0,
        ),
    )
    monkeypatch.setattr(mon, "psutil", dummy, raising=True)

    stats = mon.get_memory_stats()
    assert stats.rss_mb == 150
    assert int(stats.total_mb) == 4096
    assert int(stats.available_mb) == 3072
    # Percent reports fallback virtual memory percent when cgroup limit is unavailable
    assert abs(stats.percent - 25.0) < 1e-6


def test_log_memory_stats_and_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare psutil stubs
    dummy = SimpleNamespace(
        Process=lambda: _DummyProc(rss=200 * 1024 * 1024, percent=50.0),
        virtual_memory=lambda: _DummyVM(
            total=8 * 1024 * 1024 * 1024,
            available=5 * 1024 * 1024 * 1024,
            percent=91.0,
        ),
    )
    monkeypatch.setattr(mon, "psutil", dummy, raising=True)

    # Capture JSON logs deterministically via a dedicated handler
    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        mon.log_memory_stats(context="test_context")
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "test_context memory" in out and "rss_mb=200" in out and "total_mb=8192" in out

    # Pressure check
    assert mon.check_memory_pressure(threshold_percent=90.0) is True
    assert mon.check_memory_pressure(threshold_percent=95.0) is False


def test_log_system_info(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub psutil functions used in log_system_info
    def _cpu_count(logical: bool | None = True) -> int:
        return 8 if logical else 4

    dummy = SimpleNamespace(
        cpu_count=_cpu_count,
        virtual_memory=lambda: _DummyVM(
            total=2 * 1024 * 1024 * 1024,
            available=1 * 1024 * 1024 * 1024,
            percent=50.0,
        ),
    )
    monkeypatch.setattr(mon, "psutil", dummy, raising=True)

    logger = get_logger()
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        mon.log_system_info()
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert "system_info" in out and "cpu_logical=8" in out and "cpu_physical=4" in out
    assert "mem_total_mb=2048" in out and "mem_available_mb=1024" in out
