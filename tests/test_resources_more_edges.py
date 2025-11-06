from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_detect_cpu_cores_v2_invalid_falls_to_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V2_CPU_MAX:
            return "200000 0"  # invalid period -> fall through
        if path == res._CGROUP_V1_CFS_QUOTA:
            return "200000"
        if path == res._CGROUP_V1_CFS_PERIOD:
            return "100000"
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)
    assert res._detect_cpu_cores() == 2


def test_detect_cpu_cores_v1_zero_quota_falls_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V2_CPU_MAX:
            return "max"  # skip v2
        if path == res._CGROUP_V1_CFS_QUOTA:
            return "0"  # invalid -> fall through
        if path == res._CGROUP_V1_CFS_PERIOD:
            return "100000"
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)

    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 3, raising=True)
    assert res._detect_cpu_cores() == 3


def test_detect_cpu_cores_v2_zero_quota_uses_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V2_CPU_MAX:
            return "0 100000"  # quota <= 0 -> fall through
        if path == res._CGROUP_V1_CFS_QUOTA:
            return "300000"
        if path == res._CGROUP_V1_CFS_PERIOD:
            return "100000"
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)
    assert res._detect_cpu_cores() == 3
