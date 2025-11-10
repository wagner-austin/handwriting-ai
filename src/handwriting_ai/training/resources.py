from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from handwriting_ai.logging import get_logger


@dataclass(frozen=True)
class ResourceLimits:
    cpu_cores: int
    memory_bytes: int | None
    optimal_threads: int
    optimal_workers: int
    max_batch_size: int | None


_CGROUP_CPU_MAX: Final[Path] = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_MEM_MAX: Final[Path] = Path("/sys/fs/cgroup/memory.max")


def _read_text_file(path: Path) -> str:
    """Strict text read for resource files.

    Callers must check file existence before calling. All OSErrors are
    propagated to ensure strong error signaling at the right layer.
    """
    return path.read_text(encoding="utf-8").strip()


def _detect_cpu_cores() -> int:
    cpu_max = _read_text_file(_CGROUP_CPU_MAX)
    if cpu_max and cpu_max != "max":
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            quota = int(parts[0])
            period = int(parts[1]) or 100_000
            if quota > 0 and period > 0:
                return max(1, quota // period)
    import os

    return max(1, os.cpu_count() or 1)


def _detect_memory_limit_bytes() -> int | None:
    mem_max = _read_text_file(_CGROUP_MEM_MAX)
    if mem_max and mem_max.isdigit():
        val = int(mem_max)
        return None if val <= 0 else val
    return None


def compute_max_batch_size(memory_bytes: int | None) -> int | None:
    if memory_bytes is None:
        return None
    gb = memory_bytes / (1024 * 1024 * 1024)
    if gb < 1.0:
        return 64
    if gb < 2.0:
        return 128
    if gb < 4.0:
        return 256
    return 512


def _compute_optimal_threads(cores: int) -> int:
    c = max(1, int(cores))
    if c <= 2:
        return c
    if c <= 4:
        return c - 1
    return min(8, c)


def _compute_optimal_workers(cores: int, memory_bytes: int | None) -> int:
    """Choose DataLoader workers based on both CPU and memory.

    Conservative defaults to avoid OOM in small containers:
    - < 2 GB: 0 workers (main process loading)
    - < 4 GB: 1 worker
    - >= 4 GB: up to 2 workers, bounded by cores // 2
    """
    c = max(1, int(cores))
    if memory_bytes is None:
        # Fall back to CPU-based heuristic
        return 0 if c <= 2 else min(2, c // 2)
    gb = memory_bytes / (1024 * 1024 * 1024)
    if gb < 2.0:
        return 0
    if gb < 4.0:
        return 1 if c >= 2 else 0
    return min(2, c // 2)


def detect_resource_limits() -> ResourceLimits:
    # Existence-gated reads to avoid invalid I/O on non-container systems
    has_cpu_cg = _CGROUP_CPU_MAX.exists()
    has_mem_cg = _CGROUP_MEM_MAX.exists()

    if has_cpu_cg:
        cpu_cores = _detect_cpu_cores()
    else:
        import os as _os

        cpu_cores = max(1, _os.cpu_count() or 1)

    mem_bytes = _detect_memory_limit_bytes() if has_mem_cg else None
    optimal_threads = _compute_optimal_threads(cpu_cores)
    optimal_workers = _compute_optimal_workers(cpu_cores, mem_bytes)
    max_bs = compute_max_batch_size(mem_bytes)
    log = get_logger()
    mem_mb = int(mem_bytes // (1024 * 1024)) if isinstance(mem_bytes, int) else None
    log.info(
        "resource_limits "
        f"cpu_cores={cpu_cores} memory_mb={mem_mb} "
        f"optimal_threads={optimal_threads} optimal_workers={optimal_workers} "
        f"max_batch_size={max_bs}"
    )
    return ResourceLimits(
        cpu_cores=int(cpu_cores),
        memory_bytes=mem_bytes,
        optimal_threads=int(optimal_threads),
        optimal_workers=int(optimal_workers),
        max_batch_size=max_bs,
    )
