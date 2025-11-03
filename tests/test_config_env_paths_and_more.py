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


def test_env_app_paths_and_threads_merge() -> None:
    with tempfile.TemporaryDirectory() as td:
        env: dict[str, str] = {
            "APP__DATA_ROOT": (Path(td) / "data").as_posix(),
            "APP__ARTIFACTS_ROOT": (Path(td) / "artifacts").as_posix(),
            "APP__LOGS_ROOT": (Path(td) / "logs").as_posix(),
            "APP__THREADS": "4",
        }
        # Ensure TOML not present to avoid overriding
        env["HANDWRITING_CONFIG"] = (Path(td) / "missing.toml").as_posix()
        s = _load_with_env(env)
        assert s.app.data_root.as_posix().endswith("data")
        assert s.app.artifacts_root.as_posix().endswith("artifacts")
        assert s.app.logs_root.as_posix().endswith("logs")
        assert int(s.app.threads) == 4


def test_env_tta_true_and_false_mapping() -> None:
    with tempfile.TemporaryDirectory() as td:
        # True mapping
        env_true: dict[str, str] = {
            "DIGITS__TTA": "TrUe",
            "HANDWRITING_CONFIG": (Path(td) / "missing.toml").as_posix(),
        }
        s_true = _load_with_env(env_true)
        assert s_true.digits.tta is True
        # False mapping ("0" not in truthy set)
        env_false: dict[str, str] = {
            "DIGITS__TTA": "0",
            "HANDWRITING_CONFIG": (Path(td) / "missing.toml").as_posix(),
        }
        s_false = _load_with_env(env_false)
        assert s_false.digits.tta is False


def test_toml_digits_predict_timeout_and_visualize_kb_merge() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[digits]
predict_timeout_seconds = 7
visualize_max_kb = 64
""".strip(),
            encoding="utf-8",
        )
        s = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
        assert int(s.digits.predict_timeout_seconds) == 7
        assert int(s.digits.visualize_max_kb) == 64


def test_security_api_key_non_string_ignored_defaults_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[security]
api_key = 42
""".strip(),
            encoding="utf-8",
        )
        s = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
        assert s.security.api_key == ""
