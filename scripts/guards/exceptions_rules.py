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
    while idx < n:
        raw = lines[idx]
        if not raw.lstrip().startswith("except"):
            idx += 1
            continue
        indent = len(raw) - len(raw.lstrip(" \t"))
        next_idx, ok = _block_has_raise_or_log(lines, idx + 1, indent)
        if not ok:
            out.append(Violation(path, idx + 1, "silent-except", raw.rstrip()))
        idx = next_idx if next_idx > idx else idx + 1
    return out


def _block_has_raise_or_log(lines: list[str], start: int, indent: int) -> tuple[int, bool]:
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
    return j, bool(found_raise or found_log)
