from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from guards import RuleReport, Violation
from guards.exceptions_rules import ExceptionsRule
from guards.logging_rules import LoggingRule
from guards.typing_rules import TypingRule
from guards.util import default_file_set


@dataclass(frozen=True)
class _RunOutcome:
    reports: list[RuleReport]
    violations: list[Violation]


def _run_all(files: list[Path]) -> _RunOutcome:
    rules = [TypingRule(), LoggingRule(), ExceptionsRule()]
    reports: list[RuleReport] = []
    violations: list[Violation] = []
    for r in rules:
        res = r.run(files)
        reports.append(RuleReport(name=r.name, violations=len(res)))
        violations.extend(res)
    return _RunOutcome(reports=reports, violations=violations)


def _print_summary(outcome: _RunOutcome, verbose: bool) -> None:
    if verbose:
        print("Guard rule summary:")
        for rep in outcome.reports:
            print(f"  - {rep.name}: {rep.violations} violation(s)")
    if outcome.violations:
        print("Guard checks failed:")
        for v in outcome.violations:
            print(f"  [{v.kind}] {v.file}:{v.line_no}: {v.line}")
    else:
        print("Guard checks passed: no violations found.")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repository guard checks")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-rule summary")
    args = ap.parse_args(list(argv) if argv is not None else None)
    files = default_file_set()
    outcome = _run_all(files)
    _print_summary(outcome, verbose=bool(args.verbose))
    return 1 if outcome.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
