from __future__ import annotations

import json
import time
from pathlib import Path

from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.signature import CalibrationSignature


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
    if isinstance(v, int | float):
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as exc:
            import logging as _logging

            _logging.getLogger("handwriting_ai").error(
                "calib_parse_int_failed key=%s value=%s error=%s", key, v, exc
            )
            raise
    return default


def _get_float(d: dict[str, object], key: str, default: float) -> float:
    v = d.get(key)
    if isinstance(v, bool):
        return default
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError as exc:
            import logging as _logging

            _logging.getLogger("handwriting_ai").error(
                "calib_parse_float_failed key=%s value=%s error=%s", key, v, exc
            )
            raise
    return default


def _read_cache(path: Path) -> tuple[CalibrationSignature, CalibrationResult, float] | None:
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8")
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        import logging as _logging

        _logging.getLogger("handwriting_ai").error(
            "calib_cache_decode_failed path=%s error=%s", path, exc
        )
        raise
    root = _as_obj_dict(parsed)
    if root is None:
        raise ValueError("calibration cache root must be an object")
    sig_raw = _as_obj_dict(root.get("signature"))
    res_raw = _as_obj_dict(root.get("result"))
    ts_raw = root.get("created_at_ts")
    if sig_raw is None or res_raw is None or not isinstance(ts_raw, int | float):
        raise ValueError("calibration cache missing required fields")
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
