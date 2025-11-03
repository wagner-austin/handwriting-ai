from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Violation:
    file: Path
    line_no: int
    kind: str
    line: str


@dataclass(frozen=True)
class RuleReport:
    name: str
    violations: int


class Rule(Protocol):
    name: str

    def run(self, files: list[Path]) -> list[Violation]: ...
