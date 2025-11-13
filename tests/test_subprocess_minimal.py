"""Minimal subprocess test to see if child can log."""

import logging
import multiprocessing as mp


def child_func(msg: str) -> str:
    """Child process function."""
    from handwriting_ai.logging import init_logging

    init_logging()
    log = logging.getLogger("handwriting_ai")
    log.info(f"CHILD_START: {msg}")
    return "OK"


if __name__ == "__main__":
    from handwriting_ai.logging import init_logging

    init_logging()
    log = logging.getLogger("handwriting_ai")

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
