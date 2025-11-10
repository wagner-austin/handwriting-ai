from __future__ import annotations

import gc as _gc
from collections.abc import Iterable as _Iter
from dataclasses import dataclass
from statistics import quantiles

import torch
from torch.utils.data import DataLoader

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.dataset import DataLoaderConfig, PreprocessDataset
from handwriting_ai.training.optim import build_optimizer_and_scheduler as _build_optim
from handwriting_ai.training.safety import get_memory_guard_config
from handwriting_ai.training.train_utils import _build_model as _build_train_model


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

    it = iter(loader)
    first = next(it, None)
    if first is None:
        return 0.0, 0.0
    n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
    times: list[float] = []
    samples = 0
    start = _t.perf_counter()
    # Measure first batch
    t0 = _t.perf_counter()
    x, y = first
    samples += int(y.shape[0])
    times.append((_t.perf_counter() - t0) * 1000.0)
    seen = 1
    # Measure subsequent batches up to n_batches
    while seen < n_batches:
        nxt = next(it, None)
        if nxt is None:
            break
        t1 = _t.perf_counter()
        x2, y2 = nxt
        samples += int(y2.shape[0])
        times.append((_t.perf_counter() - t1) * 1000.0)
        seen += 1
    total_s = _t.perf_counter() - start
    p95 = max(times) if len(times) >= 2 else (times[0] if times else 0.0)
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


def _measure_training(
    ds_len: int,
    loader: _Iter[tuple[torch.Tensor, torch.Tensor]],
    k: int,
    *,
    device: torch.device,
    batch_size_hint: int,
) -> tuple[float, float, float, bool]:
    """Run k training steps and measure throughput/latency/memory.

    Returns: (samples_per_sec, p95_ms, peak_percent, exceeded_threshold)
    """
    import time as _t

    # Build a fresh small model on the chosen device and a default optimizer
    model = _build_train_model()
    model.train()
    # Model remains on CPU for calibration; move only tensors.

    # Minimal optimizer config (values do not affect memory footprint materially)
    @dataclass(frozen=True)
    class _OptimCfg:
        lr: float = 1e-3
        weight_decay: float = 0.01
        optim: str = "adamw"
        scheduler: str = "none"
        epochs: int = 1
        min_lr: float = 1e-5
        step_size: int = 10
        gamma: float = 0.5

    opt, _sch = _build_optim(model, _OptimCfg())

    try:
        # Warm-up a single batch to initialize optimizer state sizes (no exceptions)
        it = iter(loader)
        first = next(it, None)
        if first is None:
            import logging as _logging

            _logging.getLogger("handwriting_ai").info("calibration_no_samples")
            return 0.0, 0.0, 0.0, False
        x0, y0 = first
        x0 = x0.to(device)
        y0 = y0.to(device)
        opt.zero_grad(set_to_none=True)
        logits0 = model(x0)
        loss0 = torch.nn.functional.cross_entropy(logits0, y0)
        torch.autograd.backward((loss0,))
        opt.step()

        # Now measure k batches
        times: list[float] = []
        samples = 0
        exceeded = False
        peak_pct: float = 0.0
        thr = float(get_memory_guard_config().threshold_percent)
        n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
        start = _t.perf_counter()
        for seen, (x, y) in enumerate(it, start=1):
            t0 = _t.perf_counter()
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            torch.autograd.backward((loss,))
            opt.step()
            dt_ms = (_t.perf_counter() - t0) * 1000.0
            times.append(dt_ms)
            samples += int(y.shape[0])
            # Memory tracking: use configured guard threshold consistently
            from handwriting_ai.monitoring import get_memory_snapshot

            pct = float(get_memory_snapshot().cgroup_usage.percent)
            if pct > peak_pct:
                peak_pct = pct
            if pct >= thr:
                exceeded = True
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
        return sps, p95, peak_pct, exceeded
    finally:
        # Ensure allocations are collectible between attempts
        del model
        del opt
        del _sch
        _gc.collect()


def _measure_candidate(ds: PreprocessDataset, cand: Candidate, samples: int) -> CalibrationResult:
    """Measure a candidate using real training steps with binary search for safe batch size."""
    import logging as _logging

    torch.set_num_threads(int(cand.intra_threads))
    device = torch.device("cpu")

    bs_hi = max(1, int(cand.batch_size))
    bs_lo = 1
    best_bs = 1
    best_sps: float = 0.0
    best_p95: float = 0.0

    while bs_lo <= bs_hi:
        mid = (bs_lo + bs_hi) // 2
        try:
            cfg_try = DataLoaderConfig(
                batch_size=mid,
                num_workers=int(cand.num_workers),
                pin_memory=False,
                persistent_workers=bool(cand.num_workers > 0),
                prefetch_factor=2,
            )
            loader = _safe_loader(ds, cfg_try)
            sps, p95, peak_pct, exceeded = _measure_training(
                len(ds), loader, samples, device=device, batch_size_hint=mid
            )
            if not exceeded:
                best_bs, best_sps, best_p95 = mid, sps, p95
                bs_lo = mid + 1
            else:
                _logging.getLogger("handwriting_ai").info(
                    "calibration_backoff reason=mem_threshold peak_pct=%.1f bs=%d",
                    peak_pct,
                    mid,
                )
                bs_hi = mid - 1
        except (RuntimeError, MemoryError) as exc:
            _logging.getLogger("handwriting_ai").error(
                "calibration_backoff reason=exception exc=%s bs=%d error=%s",
                type(exc).__name__,
                mid,
                exc,
            )
            raise
        finally:
            _gc.collect()

    return CalibrationResult(
        intra_threads=cand.intra_threads,
        interop_threads=cand.interop_threads,
        num_workers=cand.num_workers,
        batch_size=best_bs,
        samples_per_sec=best_sps,
        p95_ms=best_p95,
    )
