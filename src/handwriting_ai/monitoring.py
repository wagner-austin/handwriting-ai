from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import psutil

from .logging import get_logger

# Cgroup v2 paths
_CGROUP_MEM_CURRENT: Path = Path("/sys/fs/cgroup/memory.current")
_CGROUP_MEM_MAX: Path = Path("/sys/fs/cgroup/memory.max")
_CGROUP_MEM_STAT: Path = Path("/sys/fs/cgroup/memory.stat")


@dataclass(frozen=True)
class CgroupMemoryUsage:
    """Cgroup-level memory usage (what the kernel OOM killer sees)."""

    usage_bytes: int
    limit_bytes: int
    percent: float


@dataclass(frozen=True)
class CgroupMemoryBreakdown:
    """Detailed memory breakdown from cgroup memory.stat."""

    anon_bytes: int
    file_bytes: int
    kernel_bytes: int
    slab_bytes: int


@dataclass(frozen=True)
class ProcessMemory:
    """Per-process memory information."""

    pid: int
    rss_bytes: int


@dataclass(frozen=True)
class MemorySnapshot:
    """Complete memory snapshot including process, cgroup, and worker data."""

    main_process: ProcessMemory
    workers: tuple[ProcessMemory, ...]
    cgroup_usage: CgroupMemoryUsage
    cgroup_breakdown: CgroupMemoryBreakdown


class MemoryMonitor(Protocol):
    """Protocol for memory monitoring implementations."""

    def get_snapshot(self) -> MemorySnapshot:
        """Capture current memory snapshot."""
        ...

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if memory usage exceeds threshold."""
        ...

    def log_snapshot(self, context: str) -> None:
        """Log memory snapshot with optional context."""
        ...


def _read_cgroup_file(path: Path) -> str:
    """Read cgroup file contents, raising on failure."""
    return path.read_text(encoding="utf-8").strip()


def _read_cgroup_int(path: Path) -> int:
    """Read cgroup file as integer, raising on failure."""
    content = _read_cgroup_file(path)
    return int(content)


def _parse_cgroup_stat(content: str) -> dict[str, int]:
    """Parse cgroup memory.stat format into key-value pairs.

    Kernel format: each line is "<key> <value>" where value is an integer.
    This parser targets the documented cgroup v2 memory.stat format.

    Skips malformed lines with logging to handle edge cases gracefully
    while maintaining visibility into parsing issues.
    """
    import logging

    logger = logging.getLogger("handwriting_ai")
    result: dict[str, int] = {}

    for line_num, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            logger.debug(
                "cgroup_stat_parse_skip line=%d reason=invalid_format expected=2_parts got=%d",
                line_num,
                len(parts),
            )
            continue

        key, value_str = parts
        try:
            value = int(value_str)
        except ValueError as exc:
            logger.error(
                "cgroup_stat_parse_skip line=%d key=%s reason=invalid_int value=%r error=%s",
                line_num,
                key,
                value_str,
                exc,
            )
            raise

        result[key] = value

    return result


def _read_cgroup_usage() -> CgroupMemoryUsage:
    """Read cgroup v2 memory usage and limit."""
    if not _CGROUP_MEM_CURRENT.exists():
        msg = "no cgroup memory files found (not in container?)"
        raise RuntimeError(msg)

    usage_bytes = _read_cgroup_int(_CGROUP_MEM_CURRENT)
    limit_content = _read_cgroup_file(_CGROUP_MEM_MAX)
    if limit_content == "max":
        msg = "cgroup memory.max is 'max' (unlimited)"
        raise RuntimeError(msg)
    limit_bytes = int(limit_content)

    percent = (float(usage_bytes) / float(limit_bytes)) * 100.0
    return CgroupMemoryUsage(
        usage_bytes=usage_bytes,
        limit_bytes=limit_bytes,
        percent=percent,
    )


def _read_cgroup_breakdown() -> CgroupMemoryBreakdown:
    """Read cgroup v2 memory breakdown from memory.stat.

    Validates that at least one core metric (anon or file) is present
    to ensure we got valid cgroup data.
    """
    import logging

    if not _CGROUP_MEM_STAT.exists():
        msg = "no cgroup memory.stat file found"
        raise RuntimeError(msg)

    stat_content = _read_cgroup_file(_CGROUP_MEM_STAT)
    stats = _parse_cgroup_stat(stat_content)

    # Validate we got at least some expected fields
    if not stats:
        msg = "cgroup memory.stat parsing produced no valid entries"
        raise RuntimeError(msg)

    # Extract required fields
    anon = stats.get("anon", 0)
    file_cache = stats.get("file", 0)
    kernel = stats.get("kernel", 0)
    slab = stats.get("slab", 0)

    # Validate at least one core metric is present
    if anon == 0 and file_cache == 0:
        logger = logging.getLogger("handwriting_ai")
        logger.warning(
            "cgroup_breakdown_missing_core_metrics anon=0 file=0 kernel=%d slab=%d fields=%s",
            kernel,
            slab,
            sorted(stats.keys()),
        )

    return CgroupMemoryBreakdown(
        anon_bytes=anon,
        file_bytes=file_cache,
        kernel_bytes=kernel,
        slab_bytes=slab,
    )


def _get_worker_processes(parent_pid: int) -> tuple[ProcessMemory, ...]:
    """Find all child processes of parent_pid and return their memory usage."""
    import logging

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
    except (OSError, ValueError, RuntimeError) as exc:
        logging.getLogger("handwriting_ai").error(
            "worker_process_lookup_failed pid=%s %s", parent_pid, exc
        )
        raise

    workers: list[ProcessMemory] = []
    for child in children:
        try:
            mem_info = child.memory_info()
            # Extract RSS by reading as object attribute
            rss_val = getattr(mem_info, "rss", 0)
            pid_val = getattr(child, "pid", 0)
            if isinstance(rss_val, int) and isinstance(pid_val, int):
                workers.append(ProcessMemory(pid=pid_val, rss_bytes=rss_val))
        except (OSError, ValueError, RuntimeError) as exc:
            logging.getLogger("handwriting_ai").error("worker_memory_read_failed %s", exc)
            raise

    return tuple(workers)


class CgroupMemoryMonitor:
    """Memory monitor using cgroup metrics (container environments)."""

    def get_snapshot(self) -> MemorySnapshot:
        """Capture complete memory snapshot including cgroup and worker data."""
        pid = os.getpid()
        proc = psutil.Process(pid)
        mem_info = proc.memory_info()
        # Extract RSS by reading as object attribute
        rss_val = getattr(mem_info, "rss", 0)
        main_rss = int(rss_val) if isinstance(rss_val, int) else 0

        main_process = ProcessMemory(pid=pid, rss_bytes=main_rss)
        workers = _get_worker_processes(pid)
        cgroup_usage = _read_cgroup_usage()
        cgroup_breakdown = _read_cgroup_breakdown()

        return MemorySnapshot(
            main_process=main_process,
            workers=workers,
            cgroup_usage=cgroup_usage,
            cgroup_breakdown=cgroup_breakdown,
        )

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if cgroup memory usage exceeds threshold."""
        cgroup = _read_cgroup_usage()
        return cgroup.percent >= threshold_percent

    def log_snapshot(self, context: str = "") -> None:
        """Log comprehensive memory snapshot with optional context prefix."""
        log = get_logger()
        snap = self.get_snapshot()
        ctx = f"{context} " if context else ""

        main_mb = snap.main_process.rss_bytes // (1024 * 1024)
        workers_mb = sum(w.rss_bytes for w in snap.workers) // (1024 * 1024)
        cgroup_usage_mb = snap.cgroup_usage.usage_bytes // (1024 * 1024)
        cgroup_limit_mb = snap.cgroup_usage.limit_bytes // (1024 * 1024)

        anon_mb = snap.cgroup_breakdown.anon_bytes // (1024 * 1024)
        file_mb = snap.cgroup_breakdown.file_bytes // (1024 * 1024)
        kernel_mb = snap.cgroup_breakdown.kernel_bytes // (1024 * 1024)
        slab_mb = snap.cgroup_breakdown.slab_bytes // (1024 * 1024)

        log.info(
            f"{ctx}memory "
            f"main_rss_mb={main_mb} workers_rss_mb={workers_mb} worker_count={len(snap.workers)} "
            f"cgroup_usage_mb={cgroup_usage_mb} cgroup_limit_mb={cgroup_limit_mb} "
            f"cgroup_pct={snap.cgroup_usage.percent:.1f} "
            f"anon_mb={anon_mb} file_mb={file_mb} kernel_mb={kernel_mb} slab_mb={slab_mb}"
        )


class SystemMemoryMonitor:
    """Memory monitor using system metrics (test/dev environments without cgroups)."""

    def __init__(self) -> None:
        # Get system memory total once for consistent "limit"
        self._system_total = int(psutil.virtual_memory().total)

    def get_snapshot(self) -> MemorySnapshot:
        """Capture basic memory snapshot using system metrics."""
        pid = os.getpid()
        proc = psutil.Process(pid)
        mem_info = proc.memory_info()
        rss_val = getattr(mem_info, "rss", 0)
        main_rss = int(rss_val) if isinstance(rss_val, int) else 0

        main_process = ProcessMemory(pid=pid, rss_bytes=main_rss)
        workers = _get_worker_processes(pid)

        # Create synthetic cgroup usage from system memory
        vm = psutil.virtual_memory()
        usage_bytes = int(vm.used)
        limit_bytes = self._system_total
        percent = (float(usage_bytes) / float(limit_bytes)) * 100.0

        cgroup_usage = CgroupMemoryUsage(
            usage_bytes=usage_bytes,
            limit_bytes=limit_bytes,
            percent=percent,
        )

        # Emulate cgroup breakdown using psutil.virtual_memory()
        # In non-cgroup environments, map system-wide memory stats to approximate categories:
        # - anon: use main process RSS (heap allocations)
        # - file: use system-wide buffers/cached (file-backed pages)
        # - kernel/slab: leave as 0 (no equivalent in psutil)
        anon_approx = main_rss
        # psutil.virtual_memory() provides buffers + cached as file-backed memory
        file_approx = int(getattr(vm, "buffers", 0) + getattr(vm, "cached", 0))

        cgroup_breakdown = CgroupMemoryBreakdown(
            anon_bytes=anon_approx,
            file_bytes=file_approx,
            kernel_bytes=0,  # Not available via psutil
            slab_bytes=0,  # Not available via psutil
        )

        return MemorySnapshot(
            main_process=main_process,
            workers=workers,
            cgroup_usage=cgroup_usage,
            cgroup_breakdown=cgroup_breakdown,
        )

    def check_pressure(self, threshold_percent: float) -> bool:
        """Return True if system memory usage exceeds threshold."""
        vm = psutil.virtual_memory()
        percent = (float(vm.used) / float(self._system_total)) * 100.0
        return percent >= threshold_percent

    def log_snapshot(self, context: str = "") -> None:
        """Log basic memory snapshot with system metrics."""
        log = get_logger()
        snap = self.get_snapshot()
        ctx = f"{context} " if context else ""

        main_mb = snap.main_process.rss_bytes // (1024 * 1024)
        workers_mb = sum(w.rss_bytes for w in snap.workers) // (1024 * 1024)
        usage_mb = snap.cgroup_usage.usage_bytes // (1024 * 1024)
        limit_mb = snap.cgroup_usage.limit_bytes // (1024 * 1024)

        log.info(
            f"{ctx}memory "
            f"main_rss_mb={main_mb} workers_rss_mb={workers_mb} worker_count={len(snap.workers)} "
            f"system_usage_mb={usage_mb} system_total_mb={limit_mb} "
            f"system_pct={snap.cgroup_usage.percent:.1f}"
        )


def _detect_cgroups_available() -> bool:
    """Detect if cgroup v2 memory files are available."""
    return _CGROUP_MEM_CURRENT.exists()


def _create_monitor() -> MemoryMonitor:
    """Create appropriate memory monitor based on environment."""
    if _detect_cgroups_available():
        return CgroupMemoryMonitor()
    return SystemMemoryMonitor()


# Module-level singleton monitor instance
_monitor: MemoryMonitor = _create_monitor()


def get_monitor() -> MemoryMonitor:
    """Get the module-level memory monitor instance."""
    return _monitor


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory snapshot using the active monitor."""
    return _monitor.get_snapshot()


def check_memory_pressure(threshold_percent: float = 90.0) -> bool:
    """Return True if memory usage exceeds threshold using the active monitor."""
    return _monitor.check_pressure(threshold_percent)


def log_memory_snapshot(*, context: str = "") -> None:
    """Log memory snapshot using the active monitor."""
    _monitor.log_snapshot(context)


def log_system_info() -> None:
    """Log system CPU and memory information at startup."""
    log = get_logger()
    cpu_logical = int(psutil.cpu_count(logical=True) or 0)
    cpu_physical_val = psutil.cpu_count(logical=False)
    cpu_physical = int(cpu_physical_val) if cpu_physical_val is not None else 0

    if _detect_cgroups_available():
        cgroup = _read_cgroup_usage()
        limit_mb = cgroup.limit_bytes // (1024 * 1024)
        log.info(
            "system_info "
            f"cpu_logical={cpu_logical} cpu_physical={cpu_physical} "
            f"cgroup_mem_limit_mb={limit_mb}"
        )
        return
    # Non-container path: use system memory metrics; propagate failures
    vm = psutil.virtual_memory()
    limit_mb = int(vm.total // (1024 * 1024))
    log.info(
        "system_info "
        f"cpu_logical={cpu_logical} cpu_physical={cpu_physical} "
        f"system_total_mb={limit_mb}"
    )


__all__ = [
    "CgroupMemoryBreakdown",
    "CgroupMemoryMonitor",
    "CgroupMemoryUsage",
    "MemoryMonitor",
    "MemorySnapshot",
    "ProcessMemory",
    "SystemMemoryMonitor",
    "check_memory_pressure",
    "get_memory_snapshot",
    "get_monitor",
    "log_memory_snapshot",
    "log_system_info",
]
