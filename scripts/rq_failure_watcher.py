from __future__ import annotations

from handwriting_ai.jobs.failure_watcher import run_from_env
from handwriting_ai.logging import init_logging


def main() -> int:  # pragma: no cover - thin wrapper
    # Initialize structured logging so INFO logs appear in deploy output
    init_logging()
    import logging as _logging

    _logging.getLogger("handwriting_ai").info("rq_failure_watcher entrypoint starting")
    run_from_env()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
