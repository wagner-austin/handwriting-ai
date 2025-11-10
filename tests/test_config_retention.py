from __future__ import annotations

import pytest

from handwriting_ai.config import (
    DigitsConfig,
    _apply_digits_retention_from_env,
    _merge_digits,
)


def test_env_retention_invalid_raises_after_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIGITS__RETENTION_KEEP_RUNS", "not-an-int")
    d0 = DigitsConfig()
    with pytest.raises(ValueError, match="invalid literal"):
        _apply_digits_retention_from_env(d0)


def test_merge_digits_retention_from_toml() -> None:
    base = DigitsConfig()
    merged = _merge_digits(base, {"retention_keep_runs": "2"})
    assert merged.retention_keep_runs == 2


def test_env_retention_valid_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIGITS__RETENTION_KEEP_RUNS", "4")
    d0 = DigitsConfig()
    d1 = _apply_digits_retention_from_env(d0)
    assert d1.retention_keep_runs == 4
