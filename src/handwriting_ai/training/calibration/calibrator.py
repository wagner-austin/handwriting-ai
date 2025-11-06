from __future__ import annotations

import logging as _logging
from dataclasses import dataclass
from pathlib import Path

from handwriting_ai.training.dataset import (
    DataLoaderConfig,
    MNISTLike,
    PreprocessDataset,
)
from handwriting_ai.training.dataset import (
    _TrainCfgProto as _AugCfgProto,
)
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.runtime import EffectiveConfig

from .cache import _read_cache, _valid_cache, _write_cache
from .candidates import Candidate, _generate_candidates
from .measure import CalibrationResult, _measure_candidate
from .signature import make_signature as _make_signature


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
    sig = _make_signature(limits)
    if not force:
        cached = _valid_cache(sig, _read_cache(cache_path), ttl_seconds)
        if cached is not None:
            return _result_to_effective(cached)

    cfg_aug: _AugCfgProto = _DummyCfg(batch_size=max(1, int(requested_batch_size)))
    ds = PreprocessDataset(train_base, cfg_aug)

    # Stage A: coarse evaluation across a compact candidate grid
    cands = _generate_candidates(limits, requested_batch_size)
    stage_a: list[CalibrationResult] = [_measure_candidate(ds, c, samples) for c in cands]
    stage_a.sort(key=lambda r: (-r.samples_per_sec, r.p95_ms))
    shortlist = stage_a[: min(3, len(stage_a))]

    # Stage B: refine top candidates with a larger sample budget to reduce noise
    samples_refine = max(1, samples * 2)
    refined: list[CalibrationResult] = []
    for r in shortlist:
        refined.append(
            _measure_candidate(
                ds,
                Candidate(
                    intra_threads=r.intra_threads,
                    interop_threads=r.interop_threads,
                    num_workers=r.num_workers,
                    batch_size=r.batch_size,
                ),
                samples_refine,
            )
        )
    refined.sort(key=lambda r: (-r.samples_per_sec, r.p95_ms))
    best = refined[0]

    # Emit a concise calibration report (top 3 + chosen)
    log = _logging.getLogger("handwriting_ai")

    def _fmt(r: CalibrationResult) -> str:
        return (
            f"threads={r.intra_threads} workers={r.num_workers} bs={r.batch_size} "
            f"sps={r.samples_per_sec:.2f} p95={r.p95_ms:.2f}"
        )

    top_str = ", ".join(_fmt(r) for r in refined[: min(3, len(refined))])
    log.info("calibration_report " f"candidates={len(cands)} top=[{top_str}] chosen=({_fmt(best)})")

    _write_cache(cache_path, sig, best)
    return _result_to_effective(best)


@dataclass(frozen=True)
class _DummyCfg:
    # Minimal adapter to satisfy _TrainCfgProto
    batch_size: int
    augment: bool = False
    aug_rotate: float = 0.0
    aug_translate: float = 0.0
    noise_prob: float = 0.0
    noise_salt_vs_pepper: float = 0.5
    dots_prob: float = 0.0
    dots_count: int = 0
    dots_size_px: int = 1
    blur_sigma: float = 0.0
    morph: str = "none"
    morph_kernel_px: int = 1
