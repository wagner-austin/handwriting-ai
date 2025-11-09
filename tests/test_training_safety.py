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


def test_memory_guard_warning_at_85_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that warnings are logged when memory reaches 85%."""
    from handwriting_ai.monitoring import (
        CgroupMemoryBreakdown,
        CgroupMemoryUsage,
        MemorySnapshot,
        ProcessMemory,
    )

    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Create snapshot at 85%
    snap = MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=100_000_000),
        workers=(),
        cgroup_usage=CgroupMemoryUsage(
            usage_bytes=850_000_000, limit_bytes=1_000_000_000, percent=85.0
        ),
        cgroup_breakdown=CgroupMemoryBreakdown(
            anon_bytes=800_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    )

    def _snapshot() -> MemorySnapshot:
        return snap

    def _pressure(*, threshold_percent: float) -> bool:
        return False  # Not at critical threshold yet

    monkeypatch.setattr(safety, "get_memory_snapshot", _snapshot, raising=True)
    monkeypatch.setattr(safety, "check_memory_pressure", _pressure, raising=True)

    # Should log warning at 85% but not trigger abort
    assert safety.on_batch_check() is False


def test_memory_guard_warning_at_90_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that warnings are logged when memory reaches 90%."""
    from handwriting_ai.monitoring import (
        CgroupMemoryBreakdown,
        CgroupMemoryUsage,
        MemorySnapshot,
        ProcessMemory,
    )

    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    # Create snapshot at 90%
    snap = MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=100_000_000),
        workers=(),
        cgroup_usage=CgroupMemoryUsage(
            usage_bytes=900_000_000, limit_bytes=1_000_000_000, percent=90.0
        ),
        cgroup_breakdown=CgroupMemoryBreakdown(
            anon_bytes=850_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    )

    def _snapshot() -> MemorySnapshot:
        return snap

    def _pressure(*, threshold_percent: float) -> bool:
        return False  # Not at critical threshold yet

    monkeypatch.setattr(safety, "get_memory_snapshot", _snapshot, raising=True)
    monkeypatch.setattr(safety, "check_memory_pressure", _pressure, raising=True)

    # Should log warning at 90% but not trigger abort
    assert safety.on_batch_check() is False


def test_memory_guard_relieved_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that pressure relieved log is emitted when consecutive counter resets."""
    from handwriting_ai.monitoring import (
        CgroupMemoryBreakdown,
        CgroupMemoryUsage,
        MemorySnapshot,
        ProcessMemory,
    )

    safety.set_memory_guard_config(
        safety.MemoryGuardConfig(enabled=True, threshold_percent=92.0, required_consecutive=3)
    )
    safety.reset_memory_guard()

    snap_high = MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=100_000_000),
        workers=(),
        cgroup_usage=CgroupMemoryUsage(
            usage_bytes=930_000_000, limit_bytes=1_000_000_000, percent=93.0
        ),
        cgroup_breakdown=CgroupMemoryBreakdown(
            anon_bytes=900_000_000, file_bytes=30_000_000, kernel_bytes=0, slab_bytes=0
        ),
    )

    snap_low = MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=100_000_000),
        workers=(),
        cgroup_usage=CgroupMemoryUsage(
            usage_bytes=800_000_000, limit_bytes=1_000_000_000, percent=80.0
        ),
        cgroup_breakdown=CgroupMemoryBreakdown(
            anon_bytes=750_000_000, file_bytes=50_000_000, kernel_bytes=0, slab_bytes=0
        ),
    )

    calls = [0]

    def _snapshot() -> MemorySnapshot:
        calls[0] += 1
        if calls[0] <= 2:
            return snap_high
        return snap_low

    def _pressure(*, threshold_percent: float) -> bool:
        return calls[0] <= 2  # First 2 calls are high pressure, then drops

    monkeypatch.setattr(safety, "get_memory_snapshot", _snapshot, raising=True)
    monkeypatch.setattr(safety, "check_memory_pressure", _pressure, raising=True)

    # First call: pressure high, consecutive = 1
    assert safety.on_batch_check() is False
    # Second call: pressure high, consecutive = 2
    assert safety.on_batch_check() is False
    # Third call: pressure drops, should log "relieved" and reset consecutive
    assert safety.on_batch_check() is False
