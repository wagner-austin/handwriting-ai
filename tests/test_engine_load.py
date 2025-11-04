from __future__ import annotations

import io
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.logging import _JsonFormatter, get_logger
from handwriting_ai.preprocess import preprocess_signature


def _make_engine_with_root(root: Path, active: str) -> InferenceEngine:
    dig = DigitsConfig(model_dir=root, active_model=active)
    s = Settings(app=AppConfig(), digits=dig, security=SecurityConfig())
    return InferenceEngine(s)


def test_try_load_active_manifest_and_state_load_failures() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a1"
        active_dir = model_dir / active
        active_dir.mkdir(parents=True, exist_ok=True)

        # 1) Missing files → early return, not ready
        eng = _make_engine_with_root(model_dir, active)
        eng.try_load_active()
        assert eng.ready is False

        # 2) Manifest parse failure → manifest_load_failed
        (active_dir / "manifest.json").write_text("not json", encoding="utf-8")
        (active_dir / "model.pt").write_bytes(b"\x00\x01bad")
        buf = io.StringIO()
        logger = get_logger()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        try:
            eng.try_load_active()
        finally:
            logger.removeHandler(handler)
        out = buf.getvalue()
        assert "manifest_load_failed" in out

        # 3) State dict load failure → state_dict_load_failed
        good_manifest = {
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
        (active_dir / "manifest.json").write_text(json.dumps(good_manifest), encoding="utf-8")
        (active_dir / "model.pt").write_bytes(b"\x00\x01bad")
        buf2 = io.StringIO()
        handler2 = logging.StreamHandler(buf2)
        handler2.setFormatter(_JsonFormatter())
        logger.addHandler(handler2)
        try:
            eng.try_load_active()
        finally:
            logger.removeHandler(handler2)
        assert "state_dict_load_failed" in buf2.getvalue()


def test_invalid_state_dict_variants_logged_and_not_ready() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a2"
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

        base_sd = build_fresh_state_dict("resnet18", 10)

        def _save_sd(d: dict[str, torch.Tensor]) -> None:
            torch.save(d, (active_dir / "model.pt").as_posix())

        cases: list[dict[str, torch.Tensor]] = []

        # Missing fc.weight
        d1 = dict(base_sd)
        d1.pop("fc.weight", None)
        cases.append(d1)
        # Wrong fc dims
        d2 = dict(base_sd)
        d2["fc.weight"] = torch.zeros((9, d2["fc.weight"].shape[1]))
        cases.append(d2)
        # Missing conv1
        d3 = dict(base_sd)
        d3.pop("conv1.weight", None)
        cases.append(d3)
        # Bad conv1 channels
        d4 = dict(base_sd)
        d4["conv1.weight"] = torch.zeros((64, 3, 3, 3))
        cases.append(d4)
        # Missing bn1
        d5 = dict(base_sd)
        d5.pop("bn1.weight", None)
        cases.append(d5)
        # Wrong fc in_features
        d6 = dict(base_sd)
        d6["fc.weight"] = torch.zeros((10, 256))
        cases.append(d6)

        for sd in cases:
            _save_sd(sd)
            eng = _make_engine_with_root(model_dir, active)
            buf = io.StringIO()
            logger = get_logger()
            h = logging.StreamHandler(buf)
            h.setFormatter(_JsonFormatter())
            logger.addHandler(h)
            try:
                eng.try_load_active()
            finally:
                logger.removeHandler(h)
            assert eng.ready is False
            assert "state_dict_invalid" in buf.getvalue()


def test_hot_reload_concurrent_access() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models"
        active = "a3"
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

        eng = _make_engine_with_root(model_dir, active)
        eng.try_load_active()
        assert eng.ready is True

        # Submit many concurrent predictions while triggering reload
        import concurrent.futures

        xs = [torch.zeros((1, 1, 28, 28), dtype=torch.float32) for _ in range(8)]
        from handwriting_ai.inference.types import PredictOutput

        futs: list[concurrent.futures.Future[PredictOutput]] = []
        for x in xs:
            futs.append(eng.submit_predict(x))

        # Overwrite manifest with a new model_id and same weights, then reload
        man2 = dict(man)
        man2["model_id"] = active + "_new"
        (active_dir / "manifest.json").write_text(json.dumps(man2), encoding="utf-8")
        eng.try_load_active()
        # Ensure all predictions completed and reload succeeded
        for f in futs:
            _ = f.result(timeout=2)
        assert eng.ready is True and eng.model_id == active + "_new"
