from __future__ import annotations

import io
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.preprocess import preprocess_signature

# Note: shutdown handler edge-case is covered indirectly elsewhere.
# This file focuses on admin upload paths.


class _FakeEngine(InferenceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._reloaded: bool = False

    def try_load_active(self) -> None:
        self._reloaded = True


def test_admin_upload_activate_true_non_active_model_does_not_reload() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        s = Settings(
            app=AppConfig(
                data_root=tmp / "data",
                artifacts_root=tmp / "artifacts",
                logs_root=tmp / "logs",
            ),
            digits=DigitsConfig(model_dir=tmp / "models", active_model="active_model"),
            security=SecurityConfig(api_key="k"),
        )
        eng = _FakeEngine(s)
        app = create_app(s, engine_provider=lambda: eng)
        client = TestClient(app)

        model_id = "other_model"
        man = {
            "schema_version": "v1.1",
            "model_id": model_id,
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        sd = build_fresh_state_dict("resnet18", 10)
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
        data = {"model_id": model_id, "activate": "true"}
        r = client.post(
            "/v1/admin/models/upload",
            headers={"X-Api-Key": "k"},
            files=files,
            data=data,
        )
        assert r.status_code == 200
        # Engine must not reload because uploaded model is not the active one
        assert eng._reloaded is False
