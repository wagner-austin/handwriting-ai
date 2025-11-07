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


def get_memory_stats() -> MemoryStats:
    """Return current process and system memory statistics.

    Uses psutil without optional imports or try/except so types remain strict.
    """
    proc = psutil.Process()
    mem_info = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    return MemoryStats(
        rss_mb=int(mem_info.rss // (1024 * 1024)),
        percent=float(proc.memory_percent()),
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
    """Return True if system memory usage exceeds threshold percent."""
    vm = psutil.virtual_memory()
    return float(vm.percent) >= float(threshold_percent)


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
