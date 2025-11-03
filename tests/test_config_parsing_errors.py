from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from handwriting_ai.config import Settings


def _load_with_env(env: dict[str, str]) -> Settings:
    old = os.environ.copy()
    try:
        os.environ.clear()
        for k, v in env.items():
            os.environ[k] = v
        return Settings.load()
    finally:
        os.environ.clear()
        for k, v in old.items():
            os.environ[k] = v


def test_invalid_toml_config_raises_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text("[app\nthreads=2", encoding="utf-8")  # unclosed table
        with pytest.raises(RuntimeError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})


def test_failed_read_config_raises_runtime() -> None:
    # Point to a directory to trigger read error
    with tempfile.TemporaryDirectory() as td, pytest.raises(RuntimeError):
        _ = _load_with_env({"HANDWRITING_CONFIG": Path(td).as_posix()})


def test_non_table_app_is_ignored() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text("app=42", encoding="utf-8")
        s = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
        # Defaults preserved
        assert int(s.app.port) == 8081 and int(s.app.threads) == 0


def test_non_table_digits_is_ignored() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text("digits=42", encoding="utf-8")
        s = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
        # Defaults preserved
        assert int(s.digits.predict_timeout_seconds) == 5


def test_app_threads_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[app]
threads = "oops"
""".strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})


def test_app_port_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[app]
port = "oops"
""".strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})


def test_digits_predict_timeout_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[digits]
predict_timeout_seconds = "oops"
""".strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})


def test_digits_visualize_max_kb_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[digits]
visualize_max_kb = "oops"
""".strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})


def test_digits_uncertain_threshold_wrong_type_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[digits]
uncertain_threshold = "oops"
""".strip(),
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
