from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.inference.manifest import ModelManifest


def test_load_state_dict_file_nested_and_flat(tmp_path: Path) -> None:
    from handwriting_ai.preprocess import preprocess_signature

    sd = build_fresh_state_dict("resnet18", 10)
    for mode in ("flat", "nested"):
        active = f"{mode}"
        active_dir = tmp_path / active
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
        if mode == "flat":
            torch.save(sd, (active_dir / "model.pt").as_posix())
        else:
            torch.save({"state_dict": sd}, (active_dir / "model.pt").as_posix())
        app_cfg = AppConfig()
        dig_cfg = DigitsConfig(model_dir=tmp_path, active_model=active)
        sec_cfg = SecurityConfig()
        s = Settings(app=app_cfg, digits=dig_cfg, security=sec_cfg)
        eng = InferenceEngine(s)
        eng.try_load_active()
        assert eng.ready is True


def test_softmax_output_shape_and_sum() -> None:
    class _ZeroModel:
        def eval(self) -> object:
            return self

        def __call__(self, x: Tensor) -> Tensor:
            b = int(x.shape[0]) if x.ndim == 4 else 1
            return torch.zeros((b, 10), dtype=torch.float32)

        def load_state_dict(self, sd: dict[str, Tensor]) -> object:
            return self

        # Satisfy TorchModel protocol
        def train(self) -> object:
            return self

        def state_dict(self) -> dict[str, Tensor]:
            return {}

        def parameters(self) -> list[torch.nn.Parameter]:
            return []

    s = Settings(app=AppConfig(), digits=DigitsConfig(), security=SecurityConfig())
    eng = InferenceEngine(s)
    eng._manifest = ModelManifest(
        schema_version="v1",
        model_id="m",
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at=datetime.now(UTC),
        preprocess_hash="v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
        val_acc=0.0,
        temperature=1.0,
    )
    eng._model = _ZeroModel()
    out = eng._predict_impl(torch.zeros((1, 1, 28, 28), dtype=torch.float32))
    assert len(out.probs) == 10 and abs(sum(out.probs) - 1.0) < 1e-6


def test_tta_changes_score_for_asymmetric_model() -> None:
    class _AsymModel:
        def eval(self) -> object:
            return self

        def __call__(self, x: Tensor) -> Tensor:
            b = int(x.shape[0])
            logits = torch.zeros((b, 10), dtype=torch.float32)
            # Make later batch entries slightly higher on class 1
            for i in range(b):
                logits[i, 1] = float(i)
            return logits

        def load_state_dict(self, sd: dict[str, Tensor]) -> object:
            return self

        # Satisfy TorchModel protocol
        def train(self) -> object:
            return self

        def state_dict(self) -> dict[str, Tensor]:
            return {}

        def parameters(self) -> list[torch.nn.Parameter]:
            return []

    s0 = Settings(app=AppConfig(), digits=DigitsConfig(tta=False), security=SecurityConfig())
    s1 = Settings(app=AppConfig(), digits=DigitsConfig(tta=True), security=SecurityConfig())
    for _s in (s0, s1):
        pass
    eng0 = InferenceEngine(s0)
    eng1 = InferenceEngine(s1)
    from handwriting_ai.preprocess import preprocess_signature

    man = ModelManifest(
        schema_version="v1",
        model_id="m",
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at=datetime.now(UTC),
        preprocess_hash=preprocess_signature(),
        val_acc=0.0,
        temperature=1.0,
    )
    eng0._manifest = man
    eng1._manifest = man
    eng0._model = _AsymModel()
    eng1._model = _AsymModel()

    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    out0 = eng0._predict_impl(x)
    out1 = eng1._predict_impl(x)
    # With rotation+shift TTA, confidence should not decrease
    assert out1.probs[1] >= out0.probs[1]
