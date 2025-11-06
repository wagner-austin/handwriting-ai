from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_detect_cpu_cores_cgroup_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V2_CPU_MAX:
            return "200000 100000"  # 2 cores
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_cpu_cores() == 2


def test_detect_cpu_cores_cgroup_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V1_CFS_QUOTA:
            return "200000"
        if path == res._CGROUP_V1_CFS_PERIOD:
            return "100000"
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_cpu_cores() == 2


def test_detect_memory_limit_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V2_MEM_MAX:
            return "1048576"
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_memory_limit_bytes() == 1048576


def test_detect_memory_limit_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_V1_MEM_LIMIT:
            return str(1024 * 1024 * 1024)
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_memory_limit_bytes() == 1024 * 1024 * 1024
