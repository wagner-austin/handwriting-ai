from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine
from handwriting_ai.preprocess import preprocess_signature


class _FakeEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._reloaded: bool = False

    def try_load_active(self) -> None:
        self._reloaded = True


def _settings(tmp: Path, api_key: str = "k") -> Settings:
    return Settings(
        app=AppConfig(
            data_root=tmp / "data",
            artifacts_root=tmp / "artifacts",
            logs_root=tmp / "logs",
        ),
        digits=DigitsConfig(model_dir=tmp / "models", active_model="mnist_resnet18_v1"),
        security=SecurityConfig(api_key=api_key),
    )


def test_admin_upload_unauthorized(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    res = client.post("/v1/admin/models/upload")
    assert res.status_code == 401


def test_admin_upload_writes_and_reloads(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    eng = _FakeEngine(s)
    app = create_app(s, engine_provider=lambda: eng)
    client = TestClient(app)

    man = {
        "schema_version": "v1.1",
        "model_id": s.digits.active_model,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
        "run_id": "t",
    }
    # Build a valid state dict buffer to satisfy strict server-side validation
    import torch

    from handwriting_ai.inference.engine import build_fresh_state_dict

    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        "model": ("model.pt", buf, "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "true"}
    res = client.post(
        "/v1/admin/models/upload",
        headers={"X-Api-Key": "k"},
        files=files,
        data=data,
    )
    assert res.status_code == 200
    assert '"ok":true' in res.text.replace(" ", "").lower()
    dest = s.digits.model_dir / s.digits.active_model
    assert (dest / "model.pt").exists()
    assert (dest / "manifest.json").exists()
    assert eng._reloaded is True


def test_admin_upload_invalid_manifest(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    files = {
        "manifest": ("manifest.json", io.BytesIO(b"not json"), "application/json"),
        "model": ("model.pt", io.BytesIO(b"pt"), "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "false"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 400


def test_admin_upload_sig_mismatch(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    # Build manifest with wrong preprocess hash
    man = {
        "schema_version": "v1.1",
        "model_id": s.digits.active_model,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "bad_hash",
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    # Use valid state dict to satisfy strict validation even when activate=false
    import torch

    from handwriting_ai.inference.engine import build_fresh_state_dict

    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        "model": ("model.pt", buf, "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "false"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 400


def test_admin_upload_model_id_mismatch(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    man = {
        "schema_version": "v1.1",
        "model_id": "other_model",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        "model": ("model.pt", io.BytesIO(b"pt"), "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "false"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 400


def test_admin_upload_no_activate_path(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    eng = _FakeEngine(s)
    app = create_app(s, engine_provider=lambda: eng)
    client = TestClient(app)
    man = {
        "schema_version": "v1.1",
        "model_id": s.digits.active_model,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        "model": ("model.pt", io.BytesIO(b"pt"), "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "false"}
    res = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert res.status_code == 200
    # Engine should not reload when activate is false
    assert eng._reloaded is False


def test_admin_upload_invalid_model_when_activate_true(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    man = {
        "schema_version": "v1.1",
        "model_id": s.digits.active_model,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        # Invalid/truncated weights; with activate=true this must be rejected
        "model": ("model.pt", io.BytesIO(b"pt"), "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "true"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 400
    txt = r.text.replace(" ", "").lower()
    assert '"code":"invalid_model"' in txt


def test_admin_upload_empty_model_when_not_activate_is_rejected(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _FakeEngine(s))
    client = TestClient(app)
    man = {
        "schema_version": "v1.1",
        "model_id": s.digits.active_model,
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    files = {
        "manifest": (
            "manifest.json",
            io.BytesIO(json.dumps(man).encode("utf-8")),
            "application/json",
        ),
        # Empty model blob should be rejected on the passive (no-activate) path
        "model": ("model.pt", io.BytesIO(b""), "application/octet-stream"),
    }
    data = {"model_id": s.digits.active_model, "activate": "false"}
    r = client.post("/v1/admin/models/upload", headers={"X-Api-Key": "k"}, files=files, data=data)
    assert r.status_code == 400
    txt = r.text.replace(" ", "").lower()
    assert '"code":"invalid_model"' in txt
