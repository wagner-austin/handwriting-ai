from __future__ import annotations

import logging
from dataclasses import dataclass

from handwriting_ai.monitoring import check_memory_pressure, get_memory_snapshot


@dataclass(frozen=True)
class MemoryGuardConfig:
    enabled: bool
    threshold_percent: float
    required_consecutive: int


_cfg: MemoryGuardConfig = MemoryGuardConfig(
    enabled=False, threshold_percent=92.0, required_consecutive=3
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
    # Check if we've crossed the abort threshold
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
    "set_memory_guard_config",
    "reset_memory_guard",
    "on_batch_check",
    "get_memory_guard_config",
]


def get_memory_guard_config() -> MemoryGuardConfig:
    return _cfg
