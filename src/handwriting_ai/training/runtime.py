from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .dataset import DataLoaderConfig
from .resources import ResourceLimits, detect_resource_limits


@dataclass(frozen=True)
class EffectiveConfig:
    intra_threads: int
    interop_threads: int | None
    batch_size: int
    loader_cfg: DataLoaderConfig


class _TrainCfgProto(Protocol):
    @property
    def threads(self) -> int: ...

    @property
    def batch_size(self) -> int: ...


def build_effective_config(cfg: _TrainCfgProto) -> tuple[EffectiveConfig, ResourceLimits]:
    limits = detect_resource_limits()
    eff_threads = int(cfg.threads) if int(cfg.threads) > 0 else int(limits.optimal_threads)
    # Clamp threads further under low-memory conditions to reduce allocator pressure
    mem_b = limits.memory_bytes
    if isinstance(mem_b, int):
        gb = mem_b / (1024 * 1024 * 1024)
        if gb < 2.0:
            eff_threads = min(eff_threads, 2)
        elif gb < 4.0:
            eff_threads = min(eff_threads, 4)
    eff_intra = eff_threads
    eff_inter = max(1, eff_intra // 2) if hasattr(torch, "set_num_interop_threads") else None
    eff_batch = int(cfg.batch_size)
    if limits.max_batch_size is not None and eff_batch > int(limits.max_batch_size):
        eff_batch = int(limits.max_batch_size)

    # Reduce prefetch under low-memory conditions to minimize steady RSS pressure
    prefetch = 2
    if isinstance(mem_b, int):
        gb2 = mem_b / (1024 * 1024 * 1024)
        if gb2 < 2.0:
            prefetch = 1
    loader_cfg = DataLoaderConfig(
        batch_size=eff_batch,
        num_workers=int(limits.optimal_workers),
        pin_memory=False,
        persistent_workers=bool(limits.optimal_workers > 0),
        prefetch_factor=prefetch,
    )
    return EffectiveConfig(eff_intra, eff_inter, eff_batch, loader_cfg), limits


def apply_threads(ec: EffectiveConfig) -> None:
    # Apply intra-op threads post-calibration; interop is set once before any parallel work
    torch.set_num_threads(int(ec.intra_threads))


__all__ = [
    "EffectiveConfig",
    "ResourceLimits",
    "apply_threads",
    "build_effective_config",
    "detect_resource_limits",
]
