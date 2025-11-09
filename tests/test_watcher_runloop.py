from __future__ import annotations

import contextlib
import time as _time
from dataclasses import dataclass

import pytest

from handwriting_ai.jobs.watcher.watcher import FailureWatcher as FWatcher


@dataclass
class _OneScan(FWatcher):
    _count: int = 0

    def scan_once(self) -> None:
        self._count += 1
        if self._count == 1:
            raise RuntimeError("scan error")


def test_run_forever_logs_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    fw = _OneScan("redis://", "digits", "digits:events")

    def _sleep(_secs: float) -> None:
        raise StopIteration

    monkeypatch.setattr(_time, "sleep", _sleep, raising=True)
    with contextlib.suppress(StopIteration):
        fw.run_forever()
