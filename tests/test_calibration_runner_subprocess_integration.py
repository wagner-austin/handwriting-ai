from __future__ import annotations

from pathlib import Path

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
