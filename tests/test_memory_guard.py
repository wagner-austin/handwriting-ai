from __future__ import annotations

from handwriting_ai.monitoring import (
    CgroupMemoryBreakdown,
    CgroupMemoryUsage,
    MemorySnapshot,
    ProcessMemory,
)
from handwriting_ai.training.safety import (
    MemoryGuardConfig,
    on_batch_check,
    reset_memory_guard,
    set_memory_guard_config,
)


def _mk_snapshot(percent: float) -> MemorySnapshot:
    usage = CgroupMemoryUsage(
        usage_bytes=int(percent * 1024 * 1024),
        limit_bytes=100 * 1024 * 1024,
        percent=percent,
    )
    br = CgroupMemoryBreakdown(anon_bytes=0, file_bytes=0, kernel_bytes=0, slab_bytes=0)
    return MemorySnapshot(
        main_process=ProcessMemory(pid=1, rss_bytes=0),
        workers=(),
        cgroup_usage=usage,
        cgroup_breakdown=br,
    )


def test_memory_guard_consecutive_threshold() -> None:
    # Configure guard: enabled with very low threshold and 2 consecutive checks
    # Using a near-zero threshold ensures real environment usage exceeds it reliably,
    # avoiding the need to patch monitoring functions.
    set_memory_guard_config(
        MemoryGuardConfig(
            enabled=True,
            threshold_percent=0.0,
            required_consecutive=2,
        )
    )
    reset_memory_guard()

    # First call: above-threshold but not enough consecutive yet
    assert on_batch_check() is False
    # Second call: second consecutive above-threshold -> abort
    assert on_batch_check() is True
    # Restore default disabled guard to avoid cross-test interference
    set_memory_guard_config(
        MemoryGuardConfig(enabled=False, threshold_percent=92.0, required_consecutive=3)
    )
    reset_memory_guard()
