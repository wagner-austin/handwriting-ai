from __future__ import annotations

import logging
from dataclasses import dataclass

from handwriting_ai.monitoring import (
    check_memory_pressure,
    get_memory_snapshot,
)

from .memory_diagnostics import record_batch_memory


@dataclass(frozen=True)
class MemoryGuardConfig:
    enabled: bool
    threshold_percent: float
    required_consecutive: int


def compute_memory_guard_config(memory_bytes: int | None) -> MemoryGuardConfig:
    """
    Compute memory guard settings based on available memory.

    Threshold tiers rationale:
    - < 1GB: 80% threshold - Critical: OOM killer aggressive, minimal headroom
    - 1-2GB: 85% threshold - Conservative: Limited buffer for optimizer state
    - 2-4GB: 88% threshold - Standard: Reasonable buffer for training overhead
    - >=4GB: 92% threshold - Relaxed: Ample headroom

    Calibration measures data loading only. Training adds ~30-50% overhead:
    - Model gradients (100% of params)
    - Optimizer state (Adam: 200% of params for momentum + variance)
    - Loss buffers, intermediate activations

    Lower thresholds for smaller containers provide safety margin.

    Args:
        memory_bytes: Available memory in bytes, or None if unlimited

    Returns:
        MemoryGuardConfig with computed threshold and required_consecutive=3
    """
    if memory_bytes is None:
        # No cgroup limit detected - use conservative default
        return MemoryGuardConfig(enabled=True, threshold_percent=88.0, required_consecutive=3)

    gb = float(memory_bytes) / (1024.0 * 1024.0 * 1024.0)

    if gb < 1.0:
        threshold = 80.0
    elif gb < 2.0:
        threshold = 85.0
    elif gb < 4.0:
        threshold = 88.0
    else:
        threshold = 92.0

    return MemoryGuardConfig(enabled=True, threshold_percent=threshold, required_consecutive=3)


_cfg: MemoryGuardConfig = MemoryGuardConfig(
    enabled=False, threshold_percent=0.0, required_consecutive=0
)
_consecutive: int = 0
_last_warning_pct: float = 0.0


def set_memory_guard_config(cfg: MemoryGuardConfig) -> None:
    global _cfg
    _cfg = cfg


def reset_memory_guard() -> None:
    global _consecutive
    global _last_warning_pct
    _consecutive = 0
    _last_warning_pct = 0.0


def on_batch_check() -> bool:
    """Return True if guard should abort due to sustained memory pressure.

    Uses a consecutive counter to avoid false positives.
    Logs progressive warnings at 85%, 90%, and threshold% for diagnostics.
    """
    global _consecutive
    global _last_warning_pct
    if not _cfg.enabled:
        return False
    snap = get_memory_snapshot()
    # Track diagnostics using the already captured snapshot to avoid duplicate sampling
    record_batch_memory(snapshot=snap)
    pct = snap.cgroup_usage.percent
    usage_mb = snap.cgroup_usage.usage_bytes // (1024 * 1024)
    limit_mb = snap.cgroup_usage.limit_bytes // (1024 * 1024)
    thr = float(_cfg.threshold_percent)
    log = logging.getLogger("handwriting_ai")
    # Progressive warnings to track memory buildup
    if pct >= 85.0 and _last_warning_pct < 85.0:
        _last_warning_pct = 85.0
        log.warning(
            "mem_pressure_approaching threshold=85.0%% current=%.1f%% usage_mb=%d limit_mb=%d "
            "guard_threshold=%.1f%%",
            pct,
            usage_mb,
            limit_mb,
            thr,
        )
    if pct >= 90.0 and _last_warning_pct < 90.0:
        _last_warning_pct = 90.0
        log.warning(
            "mem_pressure_high threshold=90.0%% current=%.1f%% usage_mb=%d limit_mb=%d "
            "guard_threshold=%.1f%%",
            pct,
            usage_mb,
            limit_mb,
            thr,
        )
    # Enforce based on configured threshold irrespective of cgroup availability.
    # Monitoring.check_memory_pressure already adapts to cgroup or system metrics.
    pressed = check_memory_pressure(threshold_percent=thr)
    if pressed:
        _consecutive += 1
        req = int(_cfg.required_consecutive)
        log.warning(
            "mem_pressure_critical consecutive=%d/%d threshold=%.1f%% current=%.1f%% "
            "usage_mb=%d limit_mb=%d",
            _consecutive,
            req,
            thr,
            pct,
            usage_mb,
            limit_mb,
        )
        return _consecutive >= req
    # Reset consecutive counter if pressure drops
    if _consecutive > 0:
        log.info("mem_pressure_relieved consecutive_reset=%d current=%.1f%%", _consecutive, pct)
    _consecutive = 0
    return False


__all__ = [
    "MemoryGuardConfig",
    "compute_memory_guard_config",
    "get_memory_guard_config",
    "on_batch_check",
    "reset_memory_guard",
    "set_memory_guard_config",
]


def get_memory_guard_config() -> MemoryGuardConfig:
    return _cfg
