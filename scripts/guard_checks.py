from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import scripts.guards.util as _util
from guards import RuleReport, Violation
from guards.exceptions_rules import ExceptionsRule
from guards.logging_rules import LoggingRule
from guards.typing_rules import TypingRule
from handwriting_ai.logging import get_logger, init_logging


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
    log = get_logger()
    if verbose:
        log.info("Guard rule summary:")
        for rep in outcome.reports:
            log.info("guard_rule name=%s violations=%d", rep.name, rep.violations)
    if outcome.violations:
        log.error("Guard checks failed:")
        for v in outcome.violations:
            # Limit line length to keep output concise
            text = v.line if len(v.line) <= 80 else v.line[:77] + "..."
            log.error(
                "guard_violation kind=%s file=%s line=%d text=%s",
                v.kind,
                v.file,
                v.line_no,
                text,
            )
    else:
        log.info("Guard checks passed: no violations found.")


def main(argv: Iterable[str] | None = None) -> int:
    init_logging()
    ap = argparse.ArgumentParser(description="Repository guard checks")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-rule summary")
    # Be resilient to extraneous argv (e.g., pytest coverage args) by ignoring unknowns
    args, _unknown = ap.parse_known_args(list(argv) if argv is not None else None)
    files = _util.default_file_set()
    outcome = _run_all(files)
    _print_summary(outcome, verbose=bool(args.verbose))
    return 1 if outcome.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
