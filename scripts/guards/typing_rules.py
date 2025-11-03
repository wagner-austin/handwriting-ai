from __future__ import annotations

import re
from pathlib import Path

from . import Violation


class TypingRule:
    name = "typing"

    _pat_any = re.compile(r"\btyping\.Any\b|\bAny\b")
    _pat_cast = re.compile(r"typing\.cast\s*\(")
    _pat_ignore = re.compile(r"#\s*type:\s*ignore(\[[^\]]+\])?")

    def run(self, files: list[Path]) -> list[Violation]:
        out: list[Violation] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for i, raw in enumerate(text.splitlines(), start=1):
                line = raw.rstrip()
                if self._pat_any.search(line):
                    out.append(Violation(f, i, "any", line))
                if self._pat_cast.search(line):
                    out.append(Violation(f, i, "cast", line))
                if self._pat_ignore.search(line):
                    out.append(Violation(f, i, "ignore", line))
        return out
