from __future__ import annotations

import logging
import os

import pytest

from handwriting_ai.logging import get_logger, init_logging


def test_env_level_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    # Preserve and restore logger handlers to avoid test interference
    logger = get_logger()
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_env = os.environ.get("HANDWRITING_LOG_LEVEL")
    try:
        # Known levels map as expected (case-insensitive)
        monkeypatch.setenv("HANDWRITING_LOG_LEVEL", "warning")
        lg = init_logging("json")
        assert lg.level == logging.WARNING

        monkeypatch.setenv("HANDWRITING_LOG_LEVEL", "CRITICAL")
        lg = init_logging("json")
        assert lg.level == logging.CRITICAL

        # Unknown levels fall back to INFO
        monkeypatch.setenv("HANDWRITING_LOG_LEVEL", "not-a-level")
        lg = init_logging("json")
        assert lg.level == logging.INFO
    finally:
        # Restore environment and logger state
        if old_env is None:
            monkeypatch.delenv("HANDWRITING_LOG_LEVEL", raising=False)
        else:
            monkeypatch.setenv("HANDWRITING_LOG_LEVEL", old_env)
        logger.setLevel(old_level)
        logger.handlers = old_handlers
