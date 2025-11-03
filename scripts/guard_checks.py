from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SRC_DIRS: tuple[str, ...] = ("src", "tests")
IGNORED_PARTS: set[str] = {
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
}


@dataclass(frozen=True)
class Violation:
    file: Path
    line_no: int
    kind: str
    line: str


def _iter_text_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            # Skip virtual envs or caches if ever in tree
            if any(part in IGNORED_PARTS for part in p.parts):
                continue
            yield p


def _scan() -> list[Violation]:
    roots = [Path(d) for d in SRC_DIRS]
    files = list(_iter_text_files(roots))

    # Patterns
    pat_any = re.compile(r'\btyping\.Any\b|\bAny\b')
    pat_cast = re.compile(r'typing\.cast\s*\(')
    # accept both '# type: ignore' and '# type: ignore[code]'
    pat_ignore = re.compile(r'#\s*type:\s*ignore(\[[^\]]+\])?')
    pat_print = re.compile(r'\bprint\s*\(')

    violations: list[Violation] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, raw in enumerate(text.splitlines(), start=1):
            line = raw.rstrip()
            # Disallow Any and typing.Any anywhere
            if pat_any.search(line):
                violations.append(Violation(f, i, "any", line))
            if pat_cast.search(line):
                violations.append(Violation(f, i, "cast", line))
            if pat_ignore.search(line):
                violations.append(Violation(f, i, "ignore", line))
            # Disallow print() in src only (logging required)
            if f.as_posix().startswith("src/") and pat_print.search(line):
                violations.append(Violation(f, i, "print", line))
    return violations


def main() -> int:
    violations = _scan()
    if not violations:
        print("Guard checks passed: no Any/cast/ignore and no print() in src.")
        return 0
    print("Guard checks failed:")
    for v in violations:
        print(f"  [{v.kind}] {v.file}:{v.line_no}: {v.line}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
