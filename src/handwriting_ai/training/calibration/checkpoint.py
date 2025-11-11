from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from handwriting_ai.training.calibration.measure import CalibrationResult


class CalibrationStage(str, Enum):
    A = "A"
    B = "B"


@dataclass(frozen=True)
class CalibrationCheckpoint:
    stage: CalibrationStage
    index: int
    results: list[CalibrationResult]
    shortlist: list[CalibrationResult] | None
    seed: int | None


def _encode_checkpoint(ck: CalibrationCheckpoint) -> dict[str, object]:
    return {
        "stage": ck.stage.value,
        "index": int(ck.index),
        "results": [
            {
                "intra_threads": int(r.intra_threads),
                "interop_threads": int(r.interop_threads)
                if r.interop_threads is not None
                else None,
                "num_workers": int(r.num_workers),
                "batch_size": int(r.batch_size),
                "samples_per_sec": float(r.samples_per_sec),
                "p95_ms": float(r.p95_ms),
            }
            for r in ck.results
        ],
        "shortlist": (
            [
                {
                    "intra_threads": int(r.intra_threads),
                    "interop_threads": int(r.interop_threads)
                    if r.interop_threads is not None
                    else None,
                    "num_workers": int(r.num_workers),
                    "batch_size": int(r.batch_size),
                    "samples_per_sec": float(r.samples_per_sec),
                    "p95_ms": float(r.p95_ms),
                }
                for r in ck.shortlist
            ]
            if ck.shortlist is not None
            else None
        ),
        "seed": (int(ck.seed) if ck.seed is not None else None),
    }


def _decode_checkpoint(d: dict[str, object]) -> CalibrationCheckpoint:
    stage_raw = d.get("stage")
    idx_raw = d.get("index")
    res_raw = d.get("results")
    shortlist_raw = d.get("shortlist")
    seed_raw = d.get("seed")

    if not isinstance(stage_raw, str) or not isinstance(idx_raw, int):
        raise ValueError("invalid checkpoint header")
    stage = CalibrationStage(stage_raw)

    def _to_res(x: object) -> CalibrationResult:
        if not isinstance(x, dict):
            raise ValueError("invalid result entry")
        # Required fields
        it = int(x["intra_threads"])  # raises on missing
        inter = x.get("interop_threads")
        inter_i = int(inter) if inter is not None else None
        nw = int(x["num_workers"])  # raises on missing
        bs = int(x["batch_size"])  # raises on missing
        sps = float(x["samples_per_sec"])  # raises on missing
        p95 = float(x["p95_ms"])  # raises on missing
        return CalibrationResult(
            intra_threads=it,
            interop_threads=inter_i,
            num_workers=nw,
            batch_size=bs,
            samples_per_sec=sps,
            p95_ms=p95,
        )

    results: list[CalibrationResult] = []
    if isinstance(res_raw, list):
        results = [_to_res(x) for x in res_raw]

    shortlist: list[CalibrationResult] | None = None
    if isinstance(shortlist_raw, list):
        shortlist = [_to_res(x) for x in shortlist_raw]

    seed: int | None = int(seed_raw) if isinstance(seed_raw, int) else None
    return CalibrationCheckpoint(
        stage=stage,
        index=int(idx_raw),
        results=results,
        shortlist=shortlist,
        seed=seed,
    )


def read_checkpoint(path: Path) -> CalibrationCheckpoint | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    data: object = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must be an object")
    return _decode_checkpoint(data)


def write_checkpoint(path: Path, ckpt: CalibrationCheckpoint) -> None:
    payload = json.dumps(_encode_checkpoint(ckpt), ensure_ascii=False, separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")


__all__ = [
    "CalibrationStage",
    "CalibrationCheckpoint",
    "read_checkpoint",
    "write_checkpoint",
]
