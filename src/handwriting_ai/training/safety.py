from __future__ import annotations

from dataclasses import dataclass

from handwriting_ai.monitoring import check_memory_pressure


@dataclass(frozen=True)
class MemoryGuardConfig:
    enabled: bool
    threshold_percent: float
    required_consecutive: int


_cfg: MemoryGuardConfig = MemoryGuardConfig(
    enabled=False, threshold_percent=95.0, required_consecutive=3
)
_consecutive: int = 0


def set_memory_guard_config(cfg: MemoryGuardConfig) -> None:
    global _cfg
    _cfg = cfg


def reset_memory_guard() -> None:
    global _consecutive
    _consecutive = 0


def on_batch_check() -> bool:
    """Return True if guard should abort due to sustained memory pressure.

    Uses a consecutive counter to avoid false positives.
    """
    global _consecutive
    if not _cfg.enabled:
        return False
    pressed = check_memory_pressure(threshold_percent=_cfg.threshold_percent)
    if pressed:
        _consecutive += 1
        return _consecutive >= int(_cfg.required_consecutive)
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
