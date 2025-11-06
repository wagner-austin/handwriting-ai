from __future__ import annotations

from collections.abc import Iterable as _Iter
from dataclasses import dataclass
from statistics import quantiles

import torch
from torch.utils.data import DataLoader

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import DataLoaderConfig, PreprocessDataset


@dataclass(frozen=True)
class CalibrationResult:
    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int
    samples_per_sec: float
    p95_ms: float


def _measure_loader(
    ds_len: int,
    loader: _Iter[tuple[torch.Tensor, torch.Tensor]],
    k: int,
    *,
    batch_size_hint: int,
) -> tuple[float, float]:
    import time as _t

    for _b in loader:
        break
    n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
    times: list[float] = []
    samples = 0
    start = _t.perf_counter()
    for seen, batch in enumerate(loader, start=1):
        t0 = _t.perf_counter()
        x, y = batch
        samples += int(y.shape[0])
        times.append((_t.perf_counter() - t0) * 1000.0)
        if seen >= n_batches:
            break
    total_s = _t.perf_counter() - start
    if len(times) >= 2:
        pcts = quantiles(times, n=20)
        p95 = pcts[18]
    else:
        p95 = times[0] if times else 0.0
    if samples <= 0:
        samples = int(batch_size_hint) * n_batches
    sps = float(samples) / total_s if total_s > 0 else 0.0
    return sps, p95


def _safe_loader(
    ds: PreprocessDataset, cfg: DataLoaderConfig
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        prefetch_factor=(int(cfg.prefetch_factor) if cfg.num_workers > 0 else None),
        persistent_workers=(bool(cfg.persistent_workers) if cfg.num_workers > 0 else False),
    )


def _measure_candidate(ds: PreprocessDataset, cand: Candidate, samples: int) -> CalibrationResult:
    # Apply only intra-op threads; interop is set once before calibration
    torch.set_num_threads(int(cand.intra_threads))

    bs_try = max(1, int(cand.batch_size))
    sps: float = 0.0
    p95: float = 0.0
    while bs_try >= 1:
        try:
            cfg_try = DataLoaderConfig(
                batch_size=bs_try,
                num_workers=int(cand.num_workers),
                pin_memory=False,
                persistent_workers=bool(cand.num_workers > 0),
                prefetch_factor=2,
            )
            loader = _safe_loader(ds, cfg_try)
            sps, p95 = _measure_loader(len(ds), loader, samples, batch_size_hint=bs_try)
            break
        except (RuntimeError, MemoryError):
            import logging as _logging

            _logging.getLogger("handwriting_ai").info("calibration_backoff")
            bs_try = bs_try // 2
    return CalibrationResult(
        intra_threads=cand.intra_threads,
        interop_threads=cand.interop_threads,
        num_workers=cand.num_workers,
        batch_size=bs_try if bs_try >= 1 else 1,
        samples_per_sec=sps,
        p95_ms=p95,
    )
