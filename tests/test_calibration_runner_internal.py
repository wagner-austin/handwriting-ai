from __future__ import annotations

from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from handwriting_ai.training.calibration.measure import CalibrationResult
from handwriting_ai.training.calibration.runner import (
    CandidateError,
    CandidateOutcome,
    SubprocessRunner,
    _emit_result_file,
    _encode_result_kv,
    _write_kv,
)


def test_runner_try_read_result_success(tmp_path: Path) -> None:
    p = tmp_path / "r.txt"
    p.write_text(
        "\n".join(
            [
                "ok=1",
                "intra_threads=2",
                "interop_threads=",
                "num_workers=1",
                "batch_size=8",
                "samples_per_sec=3.5",
                "p95_ms=7.0",
            ]
        ),
        encoding="utf-8",
    )
    out = SubprocessRunner._try_read_result(str(p), exited=True, exit_code=0)
    assert out is not None and out.ok and isinstance(out.res, CalibrationResult)


def test_runner_try_read_result_error(tmp_path: Path) -> None:
    p = tmp_path / "r2.txt"
    p.write_text("ok=0\nerror_message=boom\n", encoding="utf-8")
    out = SubprocessRunner._try_read_result(str(p), exited=True, exit_code=123)
    assert out is not None and not out.ok and out.error is not None
    assert out.error.exit_code == 123


def test_runner_wait_for_outcome_classifies_oom(tmp_path: Path) -> None:
    # No result file, process dead with exitcode 137 => classifies as OOM
    class _DeadProc(BaseProcess):
        def __init__(self) -> None:
            pass

        @property
        def exitcode(self) -> int:
            return 137

        def is_alive(self) -> bool:
            return False

    r = SubprocessRunner()
    out = r._wait_for_outcome(_DeadProc(), str(tmp_path / "missing.txt"), start=0.0, timeout_s=0.01)
    assert out.error is not None and out.error.kind == "oom"


def test_runner_wait_for_outcome_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a long-lived process and force timeout branch
    class _AliveProc(BaseProcess):
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            self._alive = False

    # Force perf_counter to return large value to trigger timeout immediately
    import time as _time

    monkeypatch.setattr(_time, "perf_counter", lambda: 999.0, raising=False)
    r = SubprocessRunner()
    out = r._wait_for_outcome(
        _AliveProc(), str(tmp_path / "missing.txt"), start=0.0, timeout_s=0.01
    )
    assert out.error is not None and out.error.kind == "timeout"


def test_runner_wait_for_outcome_inline_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _AliveProc(BaseProcess):
        def is_alive(self) -> bool:
            return True

    called = {"n": 0}

    def _ok(_: str, *, exited: bool, exit_code: int | None) -> CandidateOutcome | None:
        called["n"] += 1
        return CandidateOutcome(
            ok=True,
            res=CalibrationResult(1, None, 0, 1, 1.0, 1.0),
            error=None,
        )

    monkeypatch.setattr(SubprocessRunner, "_try_read_result", staticmethod(_ok), raising=False)
    r = SubprocessRunner()
    out = r._wait_for_outcome(_AliveProc(), str(tmp_path / "x.txt"), start=0.0, timeout_s=1.0)
    assert out.ok and out.res is not None and called["n"] >= 1


def test_runner_wait_for_outcome_inline_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _AliveProc(BaseProcess):
        def is_alive(self) -> bool:
            return True

    def _err(_: str, *, exited: bool, exit_code: int | None) -> CandidateOutcome | None:
        return CandidateOutcome(
            ok=False,
            res=None,
            error=CandidateError(kind="runtime", message="boom", exit_code=None),
        )

    monkeypatch.setattr(SubprocessRunner, "_try_read_result", staticmethod(_err), raising=False)
    r = SubprocessRunner()
    out = r._wait_for_outcome(_AliveProc(), str(tmp_path / "x.txt"), start=0.0, timeout_s=1.0)
    assert (not out.ok) and out.error is not None and out.error.kind == "runtime"


def test_runner_wait_for_outcome_dead_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DeadProc(BaseProcess):
        def is_alive(self) -> bool:
            return False

        @property
        def exitcode(self) -> int:
            return 0

    def _ok(_: str, *, exited: bool, exit_code: int | None) -> CandidateOutcome | None:
        return CandidateOutcome(ok=True, res=CalibrationResult(1, None, 0, 1, 1.0, 1.0), error=None)

    monkeypatch.setattr(SubprocessRunner, "_try_read_result", staticmethod(_ok), raising=False)
    r = SubprocessRunner()
    out = r._wait_for_outcome(_DeadProc(), str(tmp_path / "x.txt"), start=0.0, timeout_s=0.1)
    assert out.ok and out.res is not None


def test_try_read_result_access_denied(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "r.txt"
    p.write_text("ok=1\n", encoding="utf-8")

    def _deny(path: str, mode: int) -> bool:
        return False

    monkeypatch.setattr(
        "handwriting_ai.training.calibration.runner._os.access",
        _deny,
        raising=False,
    )
    out = SubprocessRunner._try_read_result(str(p), exited=False, exit_code=None)
    assert out is None


def test_try_read_result_invalid_line(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("oops\n", encoding="utf-8")
    out = SubprocessRunner._try_read_result(str(p), exited=True, exit_code=1)
    assert out is None


def test_encode_and_write_kv(tmp_path: Path) -> None:
    res = CalibrationResult(2, None, 1, 8, 3.2, 7.5)
    lines = _encode_result_kv(res)
    assert any(ln.startswith("ok=") for ln in lines)
    out = tmp_path / "x.txt"
    _write_kv(str(out), lines)
    content = out.read_text(encoding="utf-8")
    assert "intra_threads=2" in content and "batch_size=8" in content


def test_emit_result_file(tmp_path: Path) -> None:
    res = CalibrationResult(1, None, 0, 2, 1.1, 2.2)
    out = tmp_path / "emit.txt"
    _emit_result_file(str(out), res)
    txt = out.read_text(encoding="utf-8")
    assert "ok=1" in txt and "batch_size=2" in txt


def test_runner_wait_for_outcome_dead_runtime(tmp_path: Path) -> None:
    class _DeadProc(BaseProcess):
        def is_alive(self) -> bool:
            return False

        @property
        def exitcode(self) -> int:
            return 1  # non-oom, non-timeout code

    r = SubprocessRunner()
    out = r._wait_for_outcome(_DeadProc(), str(tmp_path / "missing.txt"), start=0.0, timeout_s=0.1)
    assert out.error is not None and out.error.kind == "runtime"


def test_try_read_result_unknown_ok(tmp_path: Path) -> None:
    p = tmp_path / "unknown.txt"
    p.write_text("ok=2\nfoo=bar\n", encoding="utf-8")
    out = SubprocessRunner._try_read_result(str(p), exited=True, exit_code=5)
    assert out is None
