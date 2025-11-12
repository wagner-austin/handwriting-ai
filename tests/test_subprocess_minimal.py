"""Minimal subprocess test to see if child can log."""

import logging
import multiprocessing as mp
import sys


def child_func(msg: str) -> str:
    """Child process function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,  # Force stderr
    )
    log = logging.getLogger("test")
    log.info(f"CHILD_START: {msg}")
    return "OK"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("test")

    log.info("PARENT_START: About to spawn child")

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=child_func, args=("Hello from child",))
    proc.start()
    proc.join(timeout=5)

    if proc.is_alive():
        log.info("PARENT_ERROR: Child timed out!")
        proc.terminate()
        proc.join()
    else:
        log.info(f"PARENT_DONE: Child exited with code {proc.exitcode}")
