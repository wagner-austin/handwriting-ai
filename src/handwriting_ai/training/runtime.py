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
    eff_intra = eff_threads
    eff_inter = max(1, eff_intra // 2) if hasattr(torch, "set_num_interop_threads") else None
    eff_batch = int(cfg.batch_size)
    if limits.max_batch_size is not None and eff_batch > int(limits.max_batch_size):
        eff_batch = int(limits.max_batch_size)

    loader_cfg = DataLoaderConfig(
        batch_size=eff_batch,
        num_workers=int(limits.optimal_workers),
        pin_memory=False,
        persistent_workers=bool(limits.optimal_workers > 0),
        prefetch_factor=2,
    )
    return EffectiveConfig(eff_intra, eff_inter, eff_batch, loader_cfg), limits


def apply_threads(ec: EffectiveConfig) -> None:
    torch.set_num_threads(int(ec.intra_threads))
    if hasattr(torch, "set_num_interop_threads") and ec.interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(ec.interop_threads))
        except RuntimeError as e:
            import logging as _logging

            _logging.getLogger("handwriting_ai").info(f"set_num_interop_threads_failed msg={e!s}")
