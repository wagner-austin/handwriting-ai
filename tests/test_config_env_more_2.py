from __future__ import annotations

import os
import tempfile
from pathlib import Path

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


def test_env_app_port_valid_sets_value() -> None:
    with tempfile.TemporaryDirectory() as td:
        env: dict[str, str] = {
            "APP__PORT": "8082",
            # Ensure TOML missing to avoid overrides from repo defaults
            "HANDWRITING_CONFIG": (Path(td) / "missing.toml").as_posix(),
        }
        s = _load_with_env(env)
        assert int(s.app.port) == 8082


def test_digits_env_additional_overrides() -> None:
    with tempfile.TemporaryDirectory() as td:
        env: dict[str, str] = {
            "DIGITS__ACTIVE_MODEL": "alt_model",
            "DIGITS__UNCERTAIN_THRESHOLD": "0.33",
            "DIGITS__MAX_IMAGE_MB": "3",
            "DIGITS__MAX_IMAGE_SIDE_PX": "2049",
            "HANDWRITING_CONFIG": (Path(td) / "missing.toml").as_posix(),
        }
        s = _load_with_env(env)
        assert s.digits.active_model == "alt_model"
        assert abs(float(s.digits.uncertain_threshold) - 0.33) < 1e-9
        assert int(s.digits.max_image_mb) == 3
        assert int(s.digits.max_image_side_px) == 2049


def test_security_api_key_env_override_sets_value() -> None:
    with tempfile.TemporaryDirectory() as td:
        env: dict[str, str] = {
            "SECURITY__API_KEY": "sekret",
            "HANDWRITING_CONFIG": (Path(td) / "missing.toml").as_posix(),
        }
        s = _load_with_env(env)
        assert s.security.api_key == "sekret"
