from __future__ import annotations

from pathlib import Path

import pytest

import handwriting_ai.training.resources as res


def test_detect_cpu_cores_invalid_falls_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_CPU_MAX:
            return "200000 0"  # period=0 defaults to 100_000, so 200000/100000 = 2
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)

    # When period is 0, it defaults to 100_000, so we get 200000/100000 = 2 cores
    assert res._detect_cpu_cores() == 2


def test_detect_cpu_cores_zero_quota_falls_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_CPU_MAX:
            return "0 100000"  # quota <= 0 -> fall through
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)

    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 3, raising=True)
    assert res._detect_cpu_cores() == 3


def test_detect_cpu_cores_max_falls_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def _read(path: Path) -> str | None:
        if path == res._CGROUP_CPU_MAX:
            return "max"  # skip cgroup
        return None

    monkeypatch.setattr(res, "_read_text_file", _read, raising=True)

    import os as _os

    monkeypatch.setattr(_os, "cpu_count", lambda: 2, raising=True)
    assert res._detect_cpu_cores() == 2
