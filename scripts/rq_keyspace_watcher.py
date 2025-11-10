from __future__ import annotations

import logging

from handwriting_ai.jobs.watcher.runner import run_notify_from_env
from handwriting_ai.logging import init_logging
from handwriting_ai.monitoring import log_system_info


def main() -> int:  # pragma: no cover - small CLI shim
    init_logging()
    log_system_info()
    logging.getLogger("handwriting_ai").info("rq_keyspace_watcher entrypoint starting")
    try:
        run_notify_from_env()
        return 0
    except Exception as e:
        logging.getLogger("handwriting_ai").exception("rq_keyspace_watcher_failed: %s", e)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
