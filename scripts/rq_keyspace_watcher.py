from __future__ import annotations

import logging

from handwriting_ai.jobs.watcher.runner import run_notify_from_env


def main() -> None:  # pragma: no cover - small CLI shim
    logging.getLogger("handwriting_ai").info("rq_keyspace_watcher entrypoint starting")
    run_notify_from_env()


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
