from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

SRC_DIRS: tuple[str, ...] = ("src", "tests", "scripts")
IGNORED_PARTS: set[str] = {
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "guards",  # Exclude scripts/guards/** from guard checks
}


def iter_py_files(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if any(part in IGNORED_PARTS for part in p.parts):
                continue
            out.append(p)
    return out


def default_file_set() -> list[Path]:
    roots = [Path(d) for d in SRC_DIRS]
    return iter_py_files(roots)


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
