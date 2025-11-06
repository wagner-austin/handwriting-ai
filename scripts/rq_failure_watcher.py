from __future__ import annotations

from handwriting_ai.jobs.failure_watcher import run_from_env


def main() -> int:  # pragma: no cover - thin wrapper
    run_from_env()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())

