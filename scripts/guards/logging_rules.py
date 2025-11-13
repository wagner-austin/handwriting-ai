from __future__ import annotations

import re
from pathlib import Path

from . import Violation


class LoggingRule:
    name = "logging"

    _pat_print = re.compile(r"\bprint\s*\(")
    _pat_basicconfig = re.compile(r"\blogging\.basicConfig\s*\(")

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, raw in enumerate(text.splitlines(), start=1):
                line = raw.rstrip()
                # Apply logging rules to the provided file set, independent of path
                if self._pat_print.search(line):
                    out.append(Violation(f, i, "print", line))
                if self._pat_basicconfig.search(line):
                    out.append(Violation(f, i, "basicConfig", line))
        return out
