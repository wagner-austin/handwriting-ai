"""Integration tests for resources.py to achieve 100% coverage."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_read_text_file_with_real_file() -> None:
    """Test _read_text_file with actual file I/O to hit line 32."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("  test content  \n")
        temp_path = Path(f.name)

    try:
        result = res._read_text_file(temp_path)
        assert result == "test content"
    finally:
        temp_path.unlink()


def test_detect_memory_limit_invalid_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test return None when cgroup value is invalid."""

    def _read(path: Path) -> str:
        if path == res._CGROUP_MEM_MAX:
            # Return non-digit to skip cgroup
            return "max"
        return ""

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)
    # Return None when cgroup fails isdigit() check
    assert res._detect_memory_limit_bytes() is None


def test_detect_resource_limits_with_cpu_cgroup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test cpu_cores = _detect_cpu_cores() when cgroup files exist."""
    # Create fake cgroup files
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("200000 100000", encoding="utf-8")

    # Monkey patch the constants to point to our temp files
    monkeypatch.setattr(res, "_CGROUP_CPU_MAX", cpu_max, raising=True)
    # Make other paths not exist
    monkeypatch.setattr(res, "_CGROUP_MEM_MAX", tmp_path / "nonexistent", raising=True)

    # Call detect_resource_limits
    limits = res.detect_resource_limits()
    # Should detect 2 cores from our fake file
    assert limits.cpu_cores == 2
