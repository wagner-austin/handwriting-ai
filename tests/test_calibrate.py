from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from handwriting_ai.training.calibrate import _candidate_workers, calibrate_input_pipeline
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.resources import ResourceLimits


class _TinyBase(Dataset[tuple[Image.Image, int]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), idx % 10


def test_calibrate_persists_and_reuses_cache(tmp_path: Path) -> None:
    aug = AugmentSpec(
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=0,
        dots_size_px=1,
        blur_sigma=0.0,
        morph="none",
    )
    base = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.0, fail=False),
        augment=aug,
    )
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=128 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    cache = tmp_path / "calibration.json"
    # First run writes cache
    ec1 = calibrate_input_pipeline(
        base,
        limits=limits,
        requested_batch_size=8,
        samples=2,
        cache_path=cache,
        ttl_seconds=3600,
        force=False,
    )
    assert cache.exists()
    # Change requested batch; cached result should still be reused due to signature match
    ec2 = calibrate_input_pipeline(
        base,
        limits=limits,
        requested_batch_size=32,
        samples=2,
        cache_path=cache,
        ttl_seconds=3600,
        force=False,
    )
    assert ec2.batch_size == ec1.batch_size
    assert ec2.loader_cfg.num_workers == ec1.loader_cfg.num_workers


def test_candidate_workers_enumeration() -> None:
    # Minimal sanity on worker enumeration; calibration decides, not heuristics
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=None,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=None,
    )
    ws = _candidate_workers(limits)
    assert 0 in ws and 1 in ws


def test_calibrate_force_recomputes(tmp_path: Path) -> None:
    aug = AugmentSpec(
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
        noise_prob=0.0,
        noise_salt_vs_pepper=0.5,
        dots_prob=0.0,
        dots_count=0,
        dots_size_px=1,
        blur_sigma=0.0,
        morph="none",
    )
    base = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=8, sleep_s=0.0, fail=False),
        augment=aug,
    )
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=None,
        optimal_threads=2,
        optimal_workers=1,
        max_batch_size=None,
    )
    cache = tmp_path / "cal.json"
    # Write a bogus cache that should be ignored due to force=True
    cache.write_text("{}", encoding="utf-8")
    ec = calibrate_input_pipeline(
        base,
        limits=limits,
        requested_batch_size=4,
        samples=2,
        cache_path=cache,
        ttl_seconds=1,
        force=True,
    )
    assert ec.loader_cfg.batch_size >= 1
    assert ec.loader_cfg.num_workers in (0, 1, 2)
