from __future__ import annotations

import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from handwriting_ai.api.app import create_app
from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import build_fresh_state_dict
from handwriting_ai.preprocess import preprocess_signature


def test_background_reloader_picks_up_changes() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "bg1"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        man = {
            "schema_version": "v1.1",
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

        s = Settings(
            app=AppConfig(),
            digits=DigitsConfig(model_dir=model_dir, active_model=active),
            security=SecurityConfig(),
        )
        app = create_app(s, reload_interval_seconds=0.05)
        with TestClient(app) as client:
            # initial readiness
            assert client.get("/readyz").status_code == 200

            # modify manifest id and wait for background reload
            time.sleep(0.1)
            man2 = dict(man)
            man2["model_id"] = active + "_bg"
            mpath = active_dir / "manifest.json"
            mpath.write_text(json.dumps(man2), encoding="utf-8")
            # Ensure mtime bump across platforms
            import os

            os.utime(mpath.as_posix(), None)
            # Poll briefly
            for _ in range(40):
                r = client.get("/v1/models/active")
                if active + "_bg" in r.text:
                    break
                time.sleep(0.05)
            assert active + "_bg" in client.get("/v1/models/active").text
