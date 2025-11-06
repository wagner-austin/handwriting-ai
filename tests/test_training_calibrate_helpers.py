from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from PIL import Image

import handwriting_ai.training.calibrate as cal
from handwriting_ai.training.dataset import DataLoaderConfig, MNISTLike, PreprocessDataset
from handwriting_ai.training.resources import ResourceLimits


def test_as_obj_dict_and_number_parsers() -> None:
    assert cal._as_obj_dict(123) is None
    d = cal._as_obj_dict({1: "x", "a": 7})
    assert d == {"a": 7}

    assert cal._get_int({}, "k", 5) == 5
    assert cal._get_int({"k": True}, "k", 5) == 5
    assert cal._get_int({"k": "7"}, "k", 5) == 7
    assert cal._get_int({"k": "bad"}, "k", 5) == 5

    assert cal._get_float({}, "k", 1.5) == 1.5
    assert cal._get_float({"k": False}, "k", 1.5) == 1.5
    assert cal._get_float({"k": "2.5"}, "k", 1.5) == 2.5
    assert cal._get_float({"k": "not"}, "k", 1.5) == 1.5


def test_read_cache_decode_and_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text("not-json", encoding="utf-8")
    assert cal._read_cache(p) is None

    p2 = tmp_path / "c2.json"
    p2.write_text("{}", encoding="utf-8")
    assert cal._read_cache(p2) is None

    # Non-dict JSON yields root None branch
    p3 = tmp_path / "c3.json"
    p3.write_text("[]", encoding="utf-8")
    assert cal._read_cache(p3) is None


def test_valid_cache_mismatch_and_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    sig = cal.CalibrationSignature(cpu_cores=2, mem_bytes=None, os="x", py="3", torch="t")
    res = cal.CalibrationResult(
        intra_threads=1,
        interop_threads=None,
        num_workers=0,
        batch_size=1,
        samples_per_sec=1.0,
        p95_ms=1.0,
    )
    now = 1000.0
    monkeypatch.setattr(cal, "_now_ts", lambda: now, raising=True)
    # Mismatch sig
    assert (
        cal._valid_cache(sig, (cal.CalibrationSignature(1, None, "x", "3", "t"), res, now), 10)
        is None
    )
    # Expired
    assert cal._valid_cache(sig, (sig, res, now - 3600.0), 10) is None


def test_candidate_workers_zero_cores() -> None:
    assert cal._candidate_workers(
        ResourceLimits(
            cpu_cores=0,
            memory_bytes=None,
            optimal_threads=1,
            optimal_workers=0,
            max_batch_size=None,
        )
    ) == [0]


def test_measure_candidate_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # First call to _safe_loader raises, then succeeds
    calls = {"n": 0}

    def _safe(_: object, __: DataLoaderConfig) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no memory")
        return object()

    monkeypatch.setattr(cal, "_safe_loader", _safe, raising=True)

    def _ml(
        ds_len: int,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        k: int,
        batch_size_hint: int,
    ) -> tuple[float, float]:
        return (10.0, 5.0)

    monkeypatch.setattr(cal, "_measure_loader", _ml, raising=True)

    class _Base(MNISTLike):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            img = Image.new("L", (28, 28), 0)
            return img, idx % 10

    class _Cfg:
        augment = False
        aug_rotate = 0.0
        aug_translate = 0.0
        noise_prob = 0.0
        noise_salt_vs_pepper = 0.5
        dots_prob = 0.0
        dots_count = 0
        dots_size_px = 1
        blur_sigma = 0.0
        morph = "none"
        morph_kernel_px = 1
        batch_size = 1

    ds = PreprocessDataset(_Base(), _Cfg())
    cand = cal.Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=4)
    out = cal._measure_candidate(ds, cand, samples=4)
    assert out.batch_size in (2, 4) and out.samples_per_sec >= 0.0


def test_measure_loader_paths() -> None:
    # Single short batch with zero samples triggers fallback branch and p95 fallback
    def _loader_one() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        x = torch.zeros((0, 1, 28, 28), dtype=torch.float32)
        y = torch.zeros((0,), dtype=torch.int64)
        yield x, y

    sps, p95 = cal._measure_loader(1, _loader_one(), 1, batch_size_hint=4)
    assert p95 >= 0.0 and sps >= 0.0

    # Two batches go through quantiles path
    def _loader_two() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        y = torch.ones((1,), dtype=torch.int64)
        for _ in range(2):
            yield x, y

    sps2, p952 = cal._measure_loader(4, _loader_two(), 2, batch_size_hint=2)
    assert p952 >= 0.0 and sps2 >= 0.0


def test_measure_candidate_exhausts_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    # Always raise to force while-loop to terminate via bs_try < 1 path
    def _safe(_: object, __: DataLoaderConfig) -> object:
        raise RuntimeError("no memory")

    monkeypatch.setattr(cal, "_safe_loader", _safe, raising=True)

    def _ml2(
        ds_len: int,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        k: int,
        batch_size_hint: int,
    ) -> tuple[float, float]:
        return (0.0, 0.0)

    monkeypatch.setattr(cal, "_measure_loader", _ml2, raising=True)

    class _Base(MNISTLike):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
            return Image.new("L", (28, 28), 0), idx % 10

    class _Cfg:
        augment = False
        aug_rotate = 0.0
        aug_translate = 0.0
        noise_prob = 0.0
        noise_salt_vs_pepper = 0.5
        dots_prob = 0.0
        dots_count = 0
        dots_size_px = 1
        blur_sigma = 0.0
        morph = "none"
        morph_kernel_px = 1
        batch_size = 1

    ds = PreprocessDataset(_Base(), _Cfg())
    cand = cal.Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=1)
    out = cal._measure_candidate(ds, cand, samples=1)
    assert out.batch_size >= 1


def test_measure_loader_empty_iterable() -> None:
    def _loader_empty() -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        if False:
            yield (
                torch.zeros((1, 1, 28, 28), dtype=torch.float32),
                torch.zeros((1,), dtype=torch.int64),
            )

    sps, p95 = cal._measure_loader(1, _loader_empty(), 1, batch_size_hint=1)
    assert p95 >= 0.0 and sps >= 0.0
