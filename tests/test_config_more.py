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


def test_toml_merge_app_and_digits_and_security() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            (
                "[app]\nthreads=3\nlogs_root='"
                + (Path(td) / "logs").as_posix()
                + "'\n"
                + "[digits]\ntta='yes'\nmax_image_side_px=2048\n"
                + "[security]\napi_key='x'\napi_key_enabled=true\n"
            ),
            encoding="utf-8",
        )
        env = {"HANDWRITING_CONFIG": p.as_posix()}
        s = _load_with_env(env)
        assert int(s.app.threads) == 3
        assert s.app.logs_root.as_posix().endswith("logs")
        assert s.digits.tta is True
        assert int(s.digits.max_image_side_px) == 2048
        assert s.security.api_key == "x"


def test_toml_negative_values_raise() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.toml"
        p.write_text(
            """
[app]
port = 70000
""".strip(),
            encoding="utf-8",
        )
        raised = False
        try:
            _ = _load_with_env({"HANDWRITING_CONFIG": p.as_posix()})
        except RuntimeError:
            raised = True
        assert raised is True
