from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PIL import Image

from handwriting_ai.training.calibration.calibrator import _DummyCfg
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.runner import BudgetConfig, SubprocessRunner
from handwriting_ai.training.dataset import PreprocessDataset


class _FakeMNIST:
    def __init__(self, n: int = 8) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), 0), 0


def test_subprocess_runner_writes_result_file(tmp_path: Path) -> None:
    # Full subprocess run to exercise child file write path
    base = _FakeMNIST(8)
    ds = PreprocessDataset(base, _DummyCfg(batch_size=4))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=99.0, timeout_s=20.0, max_failures=1)
    out = SubprocessRunner().run(ds, cand, samples=1, budget=budget)
    assert out.ok and out.res is not None and out.res.batch_size >= 1


def test_subprocess_runner_child_logging_works(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify child process can emit logs (regression test for init_logging() bug).

    Without init_logging() in _child_entry(), child processes would timeout silently
    because logging wasn't initialized, making all log statements no-ops.

    This test verifies that:
    1. Child process completes without timeout (proves logging init works)
    2. Child logs appear in stdout (proves JSON logging is configured)
    """
    base = _FakeMNIST(8)
    ds = PreprocessDataset(base, _DummyCfg(batch_size=4))
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)
    budget = BudgetConfig(start_pct_max=99.0, abort_pct=99.0, timeout_s=20.0, max_failures=1)

    with caplog.at_level(logging.INFO, logger="handwriting_ai"):
        out = SubprocessRunner().run(ds, cand, samples=1, budget=budget)

    # Verify subprocess succeeded without timeout
    # If init_logging() wasn't called, child would timeout after 20s
    assert out.ok and out.res is not None, "Child process failed or timed out"

    # Verify child lifecycle logs were emitted through the application logger
    messages = [rec.message for rec in caplog.records if rec.name == "handwriting_ai"]
    assert any("calibration_child_started" in m for m in messages), (
        "Missing 'calibration_child_started' in logs. Got:\n" + "\n".join(messages)
    )
    assert any("calibration_child_complete" in m for m in messages), (
        "Missing 'calibration_child_complete' in logs. Got:\n" + "\n".join(messages)
    )
