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
    pat_any = re.compile(r"\btyping\.Any\b|\bAny\b")
    pat_cast = re.compile(r"typing\.cast\s*\(")
    # accept both '# type: ignore' and '# type: ignore[code]'
    pat_ignore = re.compile(r"#\s*type:\s*ignore(\[[^\]]+\])?")
    pat_print = re.compile(r"\bprint\s*\(")
    pat_basicconfig = re.compile(r"\blogging\.basicConfig\s*\(")
    pat_suppress = re.compile(r"\bcontextlib\.suppress\s*\(")

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
            # Disallow ad-hoc logging config in app code
            if f.as_posix().startswith("src/") and pat_basicconfig.search(line):
                violations.append(Violation(f, i, "basicConfig", line))
            # Disallow explicit exception suppression in app code
            if f.as_posix().startswith("src/") and pat_suppress.search(line):
                violations.append(Violation(f, i, "suppress", line))
    violations.extend(_scan_silent_except(files))
    return violations


def _scan_silent_except(files: list[Path]) -> list[Violation]:
    out: list[Violation] = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
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
                out.append(Violation(f, idx + 1, "silent-except", lines[idx].rstrip()))
            idx = next_idx if next_idx > idx else idx + 1
    return out


def _block_has_raise_or_log(lines: list[str], start: int, indent: int) -> tuple[int, bool]:
    """Scan block starting at 'start' with greater indent than 'indent'.

    Returns (next_index_after_block, has_raise_or_log).
    """
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
