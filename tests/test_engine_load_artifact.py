from __future__ import annotations

from pathlib import Path

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict
from handwriting_ai.training.artifacts import write_artifacts


def _settings(tmp: Path, model_id: str) -> Settings:
    app = AppConfig()
    digits = DigitsConfig(model_dir=tmp, active_model=model_id)
    sec = SecurityConfig()
    return Settings(app=app, digits=digits, security=sec)


def test_engine_try_load_active_success(tmp_path: Path) -> None:
    model_id = "mnist_resnet18_v1"
    # Build a fresh state dict compatible with our model arch
    sd = build_fresh_state_dict(arch="resnet18", n_classes=10)
    # Write artifacts to tmp model dir
    out = write_artifacts(
        out_dir=tmp_path,
        model_id=model_id,
        model_state=sd,
        epochs=1,
        batch_size=1,
        lr=1e-3,
        seed=0,
        device_str="cpu",
        optim="adamw",
        scheduler="none",
        augment=False,
        test_val_acc=0.0,
    )
    assert (out / "model.pt").exists() and (out / "manifest.json").exists()
    s = _settings(tmp_path, model_id)
    eng = InferenceEngine(s)
    # Initially not ready
    assert eng.ready is False
    eng.try_load_active()
    assert eng.ready is True


def test_engine_try_load_active_bad_model_file(tmp_path: Path) -> None:
    model_id = "mnist_resnet18_v1"
    dest = tmp_path / model_id
    dest.mkdir(parents=True, exist_ok=True)
    # Write an invalid model.pt and a minimal manifest with required fields
    (dest / "model.pt").write_bytes(b"not a torch file")
    from handwriting_ai.preprocess import preprocess_signature

    manifest = (
        "{"  # minimal valid manifest v1.1
        '"schema_version":"v1.1",'
        f'"model_id":"{model_id}",'
        '"arch":"resnet18",'
        '"n_classes":10,'
        '"version":"1.0.0",'
        '"created_at":"2024-01-01T00:00:00+00:00",'
        f'"preprocess_hash":"{preprocess_signature()}",'
        '"val_acc":0.0,'
        '"temperature":1.0'
        "}"
    )
    (dest / "manifest.json").write_text(manifest, encoding="utf-8")
    s = _settings(tmp_path, model_id)
    eng = InferenceEngine(s)
    eng.try_load_active()
    # Should not be ready due to bad model file
    assert eng.ready is False
