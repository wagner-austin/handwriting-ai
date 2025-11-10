"""Coverage tests for calibration cache.py to achieve 100%."""

from __future__ import annotations

from handwriting_ai.training.calibration.cache import _valid_cache
from handwriting_ai.training.calibration.signature import CalibrationSignature


def test_valid_cache_none_returns_none() -> None:
    """Test line 143: return None when cache is None."""
    sig = CalibrationSignature(cpu_cores=4, mem_bytes=None, os="linux", py="3.11", torch="2.0")
    # Call _valid_cache with cache=None to hit line 143
    result = _valid_cache(sig, None, ttl_s=3600)
    assert result is None
