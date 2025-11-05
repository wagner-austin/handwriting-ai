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


class _RaiseReloadEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def try_load_active(self) -> None:
        raise OSError("reload failed")


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


def test_admin_upload_reload_failure_is_logged_but_succeeds(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    app = create_app(s, engine_provider=lambda: _RaiseReloadEngine(s))
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
    # Provide a valid state dict buffer to satisfy strict validation
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
    # Despite reload failure, endpoint should return 200 with response body
    assert res.status_code == 200
    assert '"ok":true' in res.text.replace(" ", "").lower()
    dest = s.digits.model_dir / s.digits.active_model
    assert (dest / "model.pt").exists()
    assert (dest / "manifest.json").exists()
