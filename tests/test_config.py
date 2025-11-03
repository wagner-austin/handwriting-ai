from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from handwriting_ai.config import Settings


def test_app_port_out_of_range_raises() -> None:
    env = os.environ.copy()
    env["APP__PORT"] = "70000"
    # Use env overrides only; ensure HANDWRITING_CONFIG does not exist
    with pytest.raises(RuntimeError):
        _ = _load_with_env(env)


def test_digits_conf_threshold_toml_mapping() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[digits]
conf_threshold = 0.42
""".strip(),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["HANDWRITING_CONFIG"] = p.as_posix()
        s = _load_with_env(env)
        assert abs(float(s.digits.uncertain_threshold) - 0.42) < 1e-6


def test_security_api_key_enabled_false_disables_key() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[security]
api_key = "secret"
api_key_enabled = false
""".strip(),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env["HANDWRITING_CONFIG"] = p.as_posix()
        s = _load_with_env(env)
        assert s.security.api_key == ""


def test_env_overrides_happy_paths() -> None:
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env["DIGITS__MODEL_DIR"] = (Path(td) / "models").as_posix()
        env["DIGITS__VISUALIZE_MAX_KB"] = "2"
        env["DIGITS__PREDICT_TIMEOUT_SECONDS"] = "1"
        # Point to a non-existent TOML so env values are not overridden by default repo config
        env["HANDWRITING_CONFIG"] = (Path(td) / "missing.toml").as_posix()
        s = _load_with_env(env)
        assert s.digits.model_dir.as_posix().endswith("models")
        assert int(s.digits.visualize_max_kb) == 2
        assert int(s.digits.predict_timeout_seconds) == 1


def _load_with_env(env: dict[str, str]) -> Settings:
    # Isolate env for Settings.load using a child process semantics via os.environ updates/reverts
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
