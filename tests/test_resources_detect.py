from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_detect_cpu_cores_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_CPU_MAX:
            return "200000 100000"  # 2 cores
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_cpu_cores() == 2


def test_detect_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_MEM_MAX:
            return "1048576"
        return None

    monkeypatch.setattr("handwriting_ai.training.resources._read_text_file", _read)
    assert res._detect_memory_limit_bytes() == 1048576
