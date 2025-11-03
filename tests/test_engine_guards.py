from __future__ import annotations

import tempfile
from pathlib import Path

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine


def _mk_engine(root: Path, active: str) -> InferenceEngine:
    app_cfg = AppConfig()
    dig_cfg = DigitsConfig(model_dir=root, active_model=active)
    sec_cfg = SecurityConfig()
    s = Settings(app=app_cfg, digits=dig_cfg, security=sec_cfg)
    return InferenceEngine(s)


def test_try_load_active_missing_manifest_or_model() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        active = "m"
        d = root / active
        d.mkdir(parents=True, exist_ok=True)
        # Missing both files
        eng = _mk_engine(root, active)
        eng.try_load_active()
        assert eng.ready is False
        # Manifest present but model missing
        (d / "manifest.json").write_text("{}", encoding="utf-8")
        eng2 = _mk_engine(root, active)
        eng2.try_load_active()
        assert eng2.ready is False
