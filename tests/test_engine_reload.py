from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.preprocess import preprocess_signature


def _make_engine_with_root(root: Path, active: str) -> InferenceEngine:
    dig = DigitsConfig(model_dir=root, active_model=active)
    s = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    return InferenceEngine(s)


def test_reload_if_changed_detects_updates() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a_reload"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        man = {
            "schema_version": "v1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (active_dir / "model.pt").as_posix())

        eng = _make_engine_with_root(model_dir, active)
        eng.try_load_active()
        assert eng.ready is True

        # Update manifest with new model_id
        man2 = dict(man)
        man2["model_id"] = active + "_v2"
        (active_dir / "manifest.json").write_text(json.dumps(man2), encoding="utf-8")

        reloaded = eng.reload_if_changed()
        assert reloaded is True and eng.model_id == active + "_v2"

        # Unchanged files path: subsequent call returns False
        assert eng.reload_if_changed() is False


def test_reload_if_changed_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not ready / no artifacts dir
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        eng = _make_engine_with_root(root, "none")
        assert eng.reload_if_changed() is False

    # Ready but missing last mtimes
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "m"
        active_dir = model_dir / "a"
        active_dir.mkdir(parents=True, exist_ok=True)
        man = {
            "schema_version": "v1",
            "model_id": "a",
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (active_dir / "model.pt").as_posix())
        eng = _make_engine_with_root(model_dir, "a")
        eng.try_load_active()
        # Simulate unknown mtimes
        eng._last_manifest_mtime = None
        eng._last_model_mtime = None
        assert eng.reload_if_changed() is False

        # OSError path
        orig_stat = Path.stat

        def _boom(self: Path) -> object:
            p = self.as_posix()
            if p.endswith("manifest.json") or p.endswith("model.pt"):
                raise OSError("nope")
            return orig_stat(self)

        monkeypatch.setattr(Path, "stat", _boom, raising=True)
        assert eng.reload_if_changed() is False


def test_try_load_active_stat_oserror_sets_last_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate OSError when reading mtimes during initial load
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a_oserr"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        man = {
            "schema_version": "v1",
            "model_id": active,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (active_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (active_dir / "model.pt").as_posix())

        orig_stat = Path.stat

        def _boom(self: Path) -> object:
            p = self.as_posix()
            if p.endswith("manifest.json") or p.endswith("model.pt"):
                raise OSError("nope")
            return orig_stat(self)

        monkeypatch.setattr(Path, "stat", _boom, raising=True)

        # Force exists to not call stat to reach the mtimes block
        def _exists(self: Path) -> bool:
            return True

        monkeypatch.setattr(Path, "exists", _exists, raising=True)
        eng = _make_engine_with_root(model_dir, active)
        eng.try_load_active()
        assert eng.ready is True
        # Internals: last mtimes are not set
        assert eng._last_manifest_mtime is None
        assert eng._last_model_mtime is None
