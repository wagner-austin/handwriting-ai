from __future__ import annotations

from dataclasses import dataclass

import psutil

from .logging import get_logger


@dataclass(frozen=True)
class MemoryStats:
    rss_mb: int
    percent: float
    available_mb: int
    total_mb: int


def _cgroup_memory_limit_bytes() -> int | None:
    """Return container memory limit in bytes if available (cgroup), else None.

    Uses the training.resources helper if present to avoid duplicating logic.
    """
    try:
        from handwriting_ai.training.resources import _detect_memory_limit_bytes as _mem_lim

        return _mem_lim()
    except (ImportError, AttributeError) as _exc:
        # Log once at debug level to satisfy guard without noise
        import logging as _logging

        _logging.getLogger("handwriting_ai").debug("mem_limit_detect_unavailable %s", _exc)
        return None


def _compute_pressure_percent(rss_bytes: int, fallback_vm_percent: float) -> float:
    limit = _cgroup_memory_limit_bytes()
    if isinstance(limit, int) and limit > 0:
        pct = (float(rss_bytes) / float(limit)) * 100.0
        return max(0.0, min(pct, 1000.0))
    return float(fallback_vm_percent)


def get_memory_stats() -> MemoryStats:
    """Return current process and system memory statistics.

    Computes percent against cgroup memory limit when available; otherwise uses
    psutil's system-wide virtual memory percent.
    """
    proc = psutil.Process()
    mem_info = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    rss_mb = int(mem_info.rss // (1024 * 1024))
    percent = _compute_pressure_percent(int(mem_info.rss), float(sys_mem.percent))
    return MemoryStats(
        rss_mb=rss_mb,
        percent=float(percent),
        available_mb=int(sys_mem.available // (1024 * 1024)),
        total_mb=int(sys_mem.total // (1024 * 1024)),
    )


def log_memory_stats(*, context: str = "") -> None:
    """Log memory statistics with optional context prefix."""
    log = get_logger()
    stats = get_memory_stats()
    ctx = f"{context} " if context else ""
    log.info(
        f"{ctx}memory rss_mb={stats.rss_mb} percent={stats.percent:.1f} "
        f"available_mb={stats.available_mb} total_mb={stats.total_mb}"
    )


def check_memory_pressure(threshold_percent: float = 90.0) -> bool:
    """Return True if memory usage exceeds threshold; prefer cgroup limit percent."""
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    pct = _compute_pressure_percent(int(proc.memory_info().rss), float(vm.percent))
    return float(pct) >= float(threshold_percent)


def log_system_info() -> None:
    """Log system CPU and memory information at startup."""
    log = get_logger()
    cpu_logical = int(psutil.cpu_count(logical=True) or 0)
    cpu_physical_val = psutil.cpu_count(logical=False)
    cpu_physical = int(cpu_physical_val) if cpu_physical_val is not None else 0
    mem = psutil.virtual_memory()
    total_mb = int(mem.total // (1024 * 1024))
    avail_mb = int(mem.available // (1024 * 1024))
    log.info(
        "system_info "
        f"cpu_logical={cpu_logical} cpu_physical={cpu_physical} "
        f"mem_total_mb={total_mb} mem_available_mb={avail_mb}"
    )


__all__ = [
    "MemoryStats",
    "get_memory_stats",
    "log_memory_stats",
    "check_memory_pressure",
    "log_system_info",
]
