from __future__ import annotations

from handwriting_ai.training.calibration.cache import (
    _as_obj_dict,
    _get_float,
    _get_int,
    _now_ts,
    _read_cache,
    _valid_cache,
    _write_cache,
)
from handwriting_ai.training.calibration.calibrator import calibrate_input_pipeline
from handwriting_ai.training.calibration.candidates import (
    Candidate,
    _candidate_threads,
    _candidate_workers,
    _generate_candidates,
)
from handwriting_ai.training.calibration.measure import (
    CalibrationResult,
    _measure_candidate,
    _measure_loader,
    _safe_loader,
)

# Facade module to keep imports stable while separating concerns.
from handwriting_ai.training.calibration.signature import CalibrationSignature

__all__ = [
    "CalibrationSignature",
    "CalibrationResult",
    "Candidate",
    "calibrate_input_pipeline",
    "_candidate_threads",
    "_candidate_workers",
    "_generate_candidates",
    "_measure_candidate",
    "_measure_loader",
    "_safe_loader",
    "_as_obj_dict",
    "_get_float",
    "_get_int",
    "_now_ts",
    "_read_cache",
    "_valid_cache",
    "_write_cache",
]
