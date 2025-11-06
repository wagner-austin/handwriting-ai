from __future__ import annotations

import json
import platform
import time
from collections.abc import Iterable as _Iter
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles

import torch
from torch.utils.data import DataLoader

from .dataset import DataLoaderConfig, MNISTLike, PreprocessDataset
from .dataset import _TrainCfgProto as _AugCfgProto
from .resources import ResourceLimits, compute_max_batch_size
from .runtime import EffectiveConfig


@dataclass(frozen=True)
class CalibrationSignature:
    cpu_cores: int
    mem_bytes: int | None
    os: str
    py: str
    torch: str


@dataclass(frozen=True)
class CalibrationResult:
    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int
    samples_per_sec: float
    p95_ms: float


@dataclass(frozen=True)
class Candidate:
    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int


def _signature(limits: ResourceLimits) -> CalibrationSignature:
    return CalibrationSignature(
        cpu_cores=int(limits.cpu_cores),
        mem_bytes=limits.memory_bytes,
        os=f"{platform.system()}-{platform.release()}",
        py=platform.python_version(),
        torch=str(torch.__version__),
    )


def _now_ts() -> float:
    return time.time()


def _as_obj_dict(x: object) -> dict[str, object] | None:
    if not isinstance(x, dict):
        return None
    out: dict[str, object] = {}
    for k, v in x.items():
        if isinstance(k, str):
            out[k] = v
    return out


def _get_int(d: dict[str, object], key: str, default: int) -> int:
    v = d.get(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, (int | float)):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            import logging as _logging

            _logging.getLogger("handwriting_ai").debug("calib_parse_int_failed")
            return default
    return default


def _get_float(d: dict[str, object], key: str, default: float) -> float:
    v = d.get(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, (int | float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            import logging as _logging

            _logging.getLogger("handwriting_ai").debug("calib_parse_float_failed")
            return default
    return default


def _read_cache(path: Path) -> tuple[CalibrationSignature, CalibrationResult, float] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        import logging as _logging

        _logging.getLogger("handwriting_ai").debug("calib_cache_read_failed")
        return None
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        import logging as _logging

        _logging.getLogger("handwriting_ai").debug("calib_cache_decode_failed")
        return None
    root = _as_obj_dict(parsed)
    if root is None:
        return None
    sig_raw = _as_obj_dict(root.get("signature"))
    res_raw = _as_obj_dict(root.get("result"))
    ts_raw = root.get("created_at_ts")
    if sig_raw is None or res_raw is None or not isinstance(ts_raw, (int | float)):
        return None
    mem_v = sig_raw.get("mem_bytes")
    mem_parsed = _get_int(sig_raw, "mem_bytes", 0) if mem_v is not None else None
    sig = CalibrationSignature(
        cpu_cores=_get_int(sig_raw, "cpu_cores", 0),
        mem_bytes=mem_parsed,
        os=str(sig_raw.get("os", "")),
        py=str(sig_raw.get("py", "")),
        torch=str(sig_raw.get("torch", "")),
    )
    inter_raw = res_raw.get("interop_threads")
    interop_val = _get_int(res_raw, "interop_threads", 0) if inter_raw is not None else None
    res = CalibrationResult(
        intra_threads=_get_int(res_raw, "intra_threads", 1),
        interop_threads=interop_val,
        num_workers=_get_int(res_raw, "num_workers", 0),
        batch_size=_get_int(res_raw, "batch_size", 1),
        samples_per_sec=_get_float(res_raw, "samples_per_sec", 0.0),
        p95_ms=_get_float(res_raw, "p95_ms", 0.0),
    )
    return sig, res, float(ts_raw)


def _write_cache(path: Path, sig: CalibrationSignature, res: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body_sig: dict[str, object] = {
        "cpu_cores": int(sig.cpu_cores),
        "mem_bytes": (int(sig.mem_bytes) if sig.mem_bytes is not None else None),
        "os": sig.os,
        "py": sig.py,
        "torch": sig.torch,
    }
    body_res: dict[str, object] = {
        "intra_threads": int(res.intra_threads),
        "interop_threads": (int(res.interop_threads) if res.interop_threads is not None else None),
        "num_workers": int(res.num_workers),
        "batch_size": int(res.batch_size),
        "samples_per_sec": float(res.samples_per_sec),
        "p95_ms": float(res.p95_ms),
    }
    body: dict[str, object] = {
        "signature": body_sig,
        "result": body_res,
        "created_at_ts": _now_ts(),
    }
    path.write_text(json.dumps(body), encoding="utf-8")


def _valid_cache(
    sig_expected: CalibrationSignature,
    cache: tuple[CalibrationSignature, CalibrationResult, float] | None,
    ttl_s: int,
) -> CalibrationResult | None:
    if cache is None:
        return None
    sig_cur, res, ts = cache
    if sig_cur != sig_expected:
        return None
    if (_now_ts() - ts) > float(ttl_s):
        return None
    return res


def _candidate_threads(limits: ResourceLimits) -> list[int]:
    a = max(1, int(limits.cpu_cores // 2))
    b = max(1, int(limits.cpu_cores))
    return sorted({a, b})


def _candidate_workers(limits: ResourceLimits) -> list[int]:
    up = max(0, int(limits.cpu_cores // 2))
    out = [n for n in (0, 1, 2) if n <= up]
    return out if out else [0]


def _generate_candidates(limits: ResourceLimits, requested_batch_size: int) -> list[Candidate]:
    cap = compute_max_batch_size(limits.memory_bytes)
    base_bs = int(requested_batch_size)
    if cap is not None:
        base_bs = min(base_bs, int(cap))
    out: list[Candidate] = []
    for intra in _candidate_threads(limits):
        inter = max(1, intra // 2) if hasattr(torch, "set_num_interop_threads") else None
        for workers in _candidate_workers(limits):
            out.append(Candidate(intra, inter, workers, base_bs))
    return out


def _measure_candidate(ds: PreprocessDataset, cand: Candidate, samples: int) -> CalibrationResult:
    torch.set_num_threads(int(cand.intra_threads))
    if cand.interop_threads is not None and hasattr(torch, "set_num_interop_threads"):
        from contextlib import suppress as _suppress

        with _suppress(RuntimeError):
            torch.set_num_interop_threads(int(cand.interop_threads))

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


def _result_to_effective(res: CalibrationResult) -> EffectiveConfig:
    return EffectiveConfig(
        intra_threads=res.intra_threads,
        interop_threads=res.interop_threads,
        batch_size=res.batch_size,
        loader_cfg=DataLoaderConfig(
            batch_size=res.batch_size,
            num_workers=res.num_workers,
            pin_memory=False,
            persistent_workers=bool(res.num_workers > 0),
            prefetch_factor=2,
        ),
    )


def _measure_loader(
    ds_len: int,
    loader: _Iter[tuple[torch.Tensor, torch.Tensor]],
    k: int,
    *,
    batch_size_hint: int,
) -> tuple[float, float]:
    import time as _t

    # warmup one batch
    for _b in loader:
        break

    # Note: DataLoader implements __len__, but for general Iterable guard with ds_len
    n_batches = max(1, min(max(1, k), max(1, ds_len // max(1, batch_size_hint))))
    times: list[float] = []
    samples = 0
    start = _t.perf_counter()
    for seen, batch in enumerate(loader, start=1):
        t0 = _t.perf_counter()
        # simulate processing by touching tensors
        x, y = batch
        samples += int(y.shape[0])
        times.append((_t.perf_counter() - t0) * 1000.0)
        if seen >= n_batches:
            break
    total_s = _t.perf_counter() - start
    # Compute p95
    if len(times) >= 2:
        pcts = quantiles(times, n=20)
        p95 = pcts[18]
    else:
        p95 = times[0] if times else 0.0
    # Estimate samples/sec based on observed batches
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


def calibrate_input_pipeline(
    train_base: MNISTLike,
    *,
    limits: ResourceLimits,
    requested_batch_size: int,
    samples: int,
    cache_path: Path,
    ttl_seconds: int,
    force: bool,
) -> EffectiveConfig:
    sig = _signature(limits)
    if not force:
        cached = _valid_cache(sig, _read_cache(cache_path), ttl_seconds)
        if cached is not None:
            return _result_to_effective(cached)

    cfg_aug: _AugCfgProto = _DummyCfg(batch_size=max(1, int(requested_batch_size)))
    ds = PreprocessDataset(train_base, cfg_aug)

    cands = _generate_candidates(limits, requested_batch_size)
    best: CalibrationResult | None = None
    for cand in cands:
        res = _measure_candidate(ds, cand, samples)
        if (
            best is None
            or (res.samples_per_sec > best.samples_per_sec)
            or (abs(res.samples_per_sec - best.samples_per_sec) < 1e-6 and res.p95_ms < best.p95_ms)
        ):
            best = res
    assert best is not None
    _write_cache(cache_path, sig, best)
    return _result_to_effective(best)


@dataclass(frozen=True)
class _DummyCfg:
    augment: bool = False
    aug_rotate: float = 0.0
    aug_translate: float = 0.0
    # Optional augmentation-related (Protocol conformance only)
    noise_prob: float = 0.0
    noise_salt_vs_pepper: float = 0.5
    dots_prob: float = 0.0
    dots_count: int = 0
    dots_size_px: int = 1
    blur_sigma: float = 0.0
    morph: str = "none"
    morph_kernel_px: int = 1
    batch_size: int = 1
