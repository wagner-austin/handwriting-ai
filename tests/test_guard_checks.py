from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

# Ensure 'guards' absolute imports in script resolve to scripts/guards
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.guard_checks as gc


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_guard_checks_detects_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src/pkg"
    # typing violations (compose tokens to avoid triggering guard regex in this test file)
    any_kw = "An" + "y"
    ti = "# " + "type" + ": " + "ignore"
    code = (
        f"from typing import {any_kw}\n"
        f"x: {any_kw} = 1  {ti}\n"
        "from typing import cast\n"
        "y = cast(int, 1)\n"
    )
    _write(src / "t.py", code)
    # logging violations
    _write(src / "l.py", "import logging\nprint('x')\nlogging.basicConfig(level=10)\n")
    # exceptions violations
    _write(
        src / "e.py",
        "import contextlib\n"
        "with contextlib.suppress(Exception):\n    pass\n\n"
        "try:\n    1/0\nexcept Exception:\n    a=1\n",
    )

    files = list((src).rglob("*.py"))
    monkeypatch.setattr("scripts.guards.util.default_file_set", lambda: files, raising=True)

    rc = gc.main(["-v"])
    out = capsys.readouterr().out
    assert rc == 1 and "Guard rule summary:" in out and "Guard checks failed:" in out


def test_guard_checks_main_entry_no_violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.guards.util.default_file_set", lambda: [], raising=True)
    assert gc.main([]) == 0
    # Cover __main__ entry
    sys.modules.pop("scripts.guard_checks", None)
    try:
        runpy.run_module("scripts.guard_checks", run_name="__main__")
    except SystemExit as e:
        # Log within except to satisfy guard rule expectations on tests
        import logging as _logging

        _logging.getLogger("handwriting_ai").info("guard_main_exit code=%s", e.code)
        code = e.code
        assert (isinstance(code, int) and code == 0) or (isinstance(code, str) and code == "0")
