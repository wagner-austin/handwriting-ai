from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import httpx
import pytest
import scripts.worker as sw

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings


class _DummyResp:
    def __init__(self) -> None:
        self.status_code = 200
        self.content = b"ok"


class _DummyClient:
    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def __enter__(self) -> _DummyClient:  # pragma: no cover - trivial
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> Literal[False]:  # pragma: no cover - trivial
        return False

    def post(
        self,
        url: str,
        headers: dict[str, str],
        data: dict[str, str],
        files: dict[str, object],
    ) -> _DummyResp:
        return _DummyResp()


def _settings(tmp: Path) -> Settings:
    app = AppConfig(
        data_root=tmp / "data",
        artifacts_root=tmp / "artifacts",
        logs_root=tmp / "logs",
    )
    dig = DigitsConfig(model_dir=tmp / "models")
    # Set retention to 2 to verify propagation
    dig = replace(dig, retention_keep_runs=2)
    sec = SecurityConfig(api_key="k")
    return Settings(app=app, digits=dig, security=sec)


def test_worker_upload_invokes_prune(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Point Settings.load to our config with retention_keep_runs=2
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: _settings(tmp_path)))
    # Provide required env for upload
    monkeypatch.setenv("HANDWRITING_API_URL", "http://api.example")
    monkeypatch.setenv("HANDWRITING_API_KEY", "k")
    # Stub http client
    monkeypatch.setattr(httpx, "Client", _DummyClient)

    # Prepare a model directory with files
    model_dir = tmp_path / "artifacts" / "digits" / "models" / "m1"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.pt").write_bytes(b"pt")
    man = {
        "schema_version": "v1.1",
        "model_id": "m1",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": "2025-01-01T00:00:00+00:00",
        "preprocess_hash": "h",
        "val_acc": 0.9,
        "temperature": 1.0,
    }
    (model_dir / "manifest.json").write_text(json.dumps(man), encoding="utf-8")

    called: dict[str, object] = {}

    def _prune_probe(dir_path: Path, keep: int) -> list[Path]:
        called["dir"] = dir_path
        called["keep"] = keep
        return []

    monkeypatch.setattr(sw, "prune_model_artifacts", _prune_probe)

    # Invoke upload helper (will hit our dummy client and then prune)
    sw._maybe_upload_artifacts(model_dir, "m1")

    assert called.get("dir") == model_dir
    assert called.get("keep") == 2
