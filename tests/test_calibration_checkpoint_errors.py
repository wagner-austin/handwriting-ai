from __future__ import annotations

from pathlib import Path

import pytest

from handwriting_ai.training.calibration.checkpoint import (
    CalibrationCheckpoint,
    CalibrationStage,
    read_checkpoint,
    write_checkpoint,
)
from handwriting_ai.training.calibration.measure import CalibrationResult


def test_checkpoint_decode_invalid_header(tmp_path: Path) -> None:
    p = tmp_path / "bad1.json"
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        read_checkpoint(p)


def test_checkpoint_missing_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "nope.json"
    assert read_checkpoint(p) is None


def test_checkpoint_decode_invalid_result_entry(tmp_path: Path) -> None:
    p = tmp_path / "bad2.json"
    # stage present, index ok, results contains a non-dict
    p.write_text('{"stage":"A","index":0,"results":[1]}', encoding="utf-8")
    with pytest.raises(ValueError):
        read_checkpoint(p)


def test_checkpoint_roundtrip_with_shortlist(tmp_path: Path) -> None:
    res1 = CalibrationResult(
        intra_threads=1,
        interop_threads=None,
        num_workers=0,
        batch_size=4,
        samples_per_sec=1.23,
        p95_ms=4.56,
    )
    res2 = CalibrationResult(
        intra_threads=2,
        interop_threads=None,
        num_workers=1,
        batch_size=8,
        samples_per_sec=2.34,
        p95_ms=5.67,
    )
    ck = CalibrationCheckpoint(
        stage=CalibrationStage.B,
        index=2,
        results=[res1],
        shortlist=[res2],
        seed=42,
    )
    p = tmp_path / "ok.json"
    write_checkpoint(p, ck)
    ck2 = read_checkpoint(p)
    assert ck2 is not None and ck2.stage == CalibrationStage.B
    assert len(ck2.results) == 1 and ck2.results[0].batch_size == 4
    assert ck2.shortlist is not None and ck2.shortlist[0].batch_size == 8


def test_checkpoint_decode_minimal_fields(tmp_path: Path) -> None:
    # No results/shortlist/seed present
    p = tmp_path / "min.json"
    p.write_text('{"stage":"A","index":0}', encoding="utf-8")
    ck = read_checkpoint(p)
    assert ck is not None and ck.stage == CalibrationStage.A and ck.index == 0
    assert ck.results == [] and ck.shortlist is None and ck.seed is None


def test_checkpoint_decode_nondict_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad_list.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_checkpoint(p)


def test_checkpoint_write_creates_parent_dirs(tmp_path: Path) -> None:
    # Use nested path that doesn't exist
    nested = tmp_path / "a" / "b" / "ck.json"
    res = CalibrationResult(1, None, 0, 1, 2.0, 3.0)
    ck = CalibrationCheckpoint(
        stage=CalibrationStage.A, index=0, results=[res], shortlist=None, seed=None
    )
    write_checkpoint(nested, ck)
    assert nested.exists()
    ck2 = read_checkpoint(nested)
    assert ck2 is not None and ck2.index == 0 and len(ck2.results) == 1
