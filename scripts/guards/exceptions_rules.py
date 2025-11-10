from __future__ import annotations

import re
from pathlib import Path

from . import Violation
from .util import read_lines


class ExceptionsRule:
    name = "exceptions"

    _pat_suppress = re.compile(r"\bcontextlib\.suppress\s*\(")

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for f in files:
            lines = read_lines(f)
            in_src = f.as_posix().startswith("src/")
            for i, raw in enumerate(lines, start=1):
                line = raw.rstrip()
                if in_src and self._pat_suppress.search(line):
                    out.append(Violation(f, i, "suppress", line))
            out.extend(_scan_silent_excepts(f, lines))
        return out


def _scan_silent_excepts(path: Path, lines: list[str]) -> list[Violation]:
    out: list[Violation] = []
    n = len(lines)
    idx = 0
    path_str = path.as_posix()
    in_src = path_str.startswith("src/")

    while idx < n:
        raw = lines[idx]
        if not raw.lstrip().startswith("except"):
            idx += 1
            continue
        indent = len(raw) - len(raw.lstrip(" \t"))
        next_idx, found_raise, found_log = _block_signals(lines, idx + 1, indent)
        func_name = _enclosing_function_name(lines, idx, indent)
        # Enforce "must raise in except" across source files by default,
        # but allow specific functions/modules to log-and-continue when
        # explicitly permitted by policy below.
        strict = bool(in_src and (func_name != "run_forever"))
        ok = found_raise if strict else (found_raise or found_log)
        if not ok:
            out.append(Violation(path, idx + 1, "silent-except", raw.rstrip()))
        idx = next_idx if next_idx > idx else idx + 1
    return out


def _block_signals(lines: list[str], start: int, indent: int) -> tuple[int, bool, bool]:
    n = len(lines)
    j = start
    found_raise = False
    found_log = False
    while j < n:
        body = lines[j]
        if not body.strip():
            j += 1
            continue
        body_indent = len(body) - len(body.lstrip(" \t"))
        if body_indent <= indent:
            break
        b = body.strip()
        if not b.startswith("#"):
            if "raise" in b:
                found_raise = True
            if ("logger." in b and "(") or "logging.getLogger" in b:
                found_log = True
        j += 1
    return j, found_raise, found_log


def _enclosing_function_name(lines: list[str], idx: int, indent: int) -> str | None:
    j = idx - 1
    while j >= 0:
        raw = lines[j]
        if not raw.strip():
            j -= 1
            continue
        raw_indent = len(raw) - len(raw.lstrip(" \t"))
        if raw.lstrip().startswith("def ") and raw_indent <= indent:
            sig = raw.strip()
            return sig[4 : sig.find("(")].strip() if "(" in sig else sig[4:].strip()
        j -= 1
    return None
