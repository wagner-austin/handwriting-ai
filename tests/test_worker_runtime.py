from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import scripts.worker as sw
from PIL import Image

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.training.mnist_train import TrainConfig


class _FakeMNIST:
    def __init__(self, root: str, train: bool, download: bool) -> None:
        self._root = root
        self._train = train
        self._download = download

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        img = Image.new("L", (28, 28), 0)
        return img, idx % 10


def _base_cfg(tmp: Path, *, augment: bool = True) -> TrainConfig:
    return TrainConfig(
        data_root=tmp / "data_in",
        out_dir=tmp / "out_in",
        model_id="mnist_resnet18_v1",
        epochs=1,
        batch_size=4,
        lr=1e-3,
        weight_decay=1e-2,
        seed=0,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=augment,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def _settings(tmp: Path) -> Settings:
    app = AppConfig(data_root=tmp / "vol" / "data", artifacts_root=tmp / "vol" / "artifacts")
    dig = DigitsConfig()
    sec = SecurityConfig()
    return Settings(app=app, digits=dig, security=sec)


def _no_upload(model_dir: Path, model_id: str) -> None:
    _ = (model_dir, model_id)
    return


def test_real_run_training_injects_paths_and_aug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Patch Settings.load to return our volume layout (patch the class directly)
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: _settings(tmp_path)))
    # Replace torchvision MNIST with a tiny fake
    monkeypatch.setattr(sw, "MNIST", _FakeMNIST)
    # Spy train_with_config to capture effective cfg
    captured_cfg: list[TrainConfig] = []

    def _train_spy(cfg: TrainConfig, bases: tuple[object, object]) -> Path:
        _ = bases
        captured_cfg.append(cfg)
        out = tmp_path / "vol" / "artifacts" / "digits" / "models" / cfg.model_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(sw, "train_with_config", _train_spy)

    # Do not attempt HTTP upload
    def _no_upload(model_dir: Path, model_id: str) -> None:
        _ = (model_dir, model_id)
        return

    monkeypatch.setattr(sw, "_maybe_upload_artifacts", _no_upload)

    cfg0 = _base_cfg(tmp_path, augment=True)
    out_dir = sw._real_run_training(cfg0)

    # Paths resolved from Settings
    assert out_dir == (tmp_path / "vol" / "artifacts" / "digits" / "models" / cfg0.model_id)
    eff = captured_cfg[0]
    assert eff.data_root == (tmp_path / "vol" / "data" / "mnist")
    assert eff.out_dir == (tmp_path / "vol" / "artifacts" / "digits" / "models")
    # Augmentation defaults injected
    assert eff.augment is True
    assert eff.noise_prob == pytest.approx(0.15)
    assert eff.dots_prob == pytest.approx(0.20)
    assert eff.dots_count == 3
    assert eff.dots_size_px == 2
    assert eff.progress_every_epochs >= 1
    # Calibration is enabled by default on worker runs
    assert eff.calibrate is True


def test_real_run_training_respects_non_default_aug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: _settings(tmp_path)))
    monkeypatch.setattr(sw, "MNIST", _FakeMNIST)
    captured_cfg: list[TrainConfig] = []

    def _train_spy(cfg: TrainConfig, bases: tuple[object, object]) -> Path:
        _ = bases
        captured_cfg.append(cfg)
        out = tmp_path / "vol" / "artifacts" / "digits" / "models" / cfg.model_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(sw, "train_with_config", _train_spy)
    monkeypatch.setattr(sw, "_maybe_upload_artifacts", _no_upload)

    cfg0 = _base_cfg(tmp_path, augment=True)
    cfg0 = replace(
        cfg0,
        noise_prob=0.33,
        dots_prob=0.11,
        dots_count=5,
        dots_size_px=3,
        progress_every_epochs=2,
    )
    _ = sw._real_run_training(cfg0)
    eff = captured_cfg[0]
    # Ensure worker did not override caller-provided non-defaults
    assert eff.noise_prob == pytest.approx(0.33)
    assert eff.dots_prob == pytest.approx(0.11)
    assert eff.dots_count == 5
    assert eff.dots_size_px == 3
    assert eff.progress_every_epochs == 2


def test_real_run_training_no_aug_leaves_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Settings, "load", staticmethod(lambda: _settings(tmp_path)))
    monkeypatch.setattr(sw, "MNIST", _FakeMNIST)
    captured_cfg: list[TrainConfig] = []

    def _train_spy(cfg: TrainConfig, bases: tuple[object, object]) -> Path:
        _ = bases
        captured_cfg.append(cfg)
        out = tmp_path / "vol" / "artifacts" / "digits" / "models" / cfg.model_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(sw, "train_with_config", _train_spy)
    monkeypatch.setattr(sw, "_maybe_upload_artifacts", _no_upload)

    cfg0 = _base_cfg(tmp_path, augment=False)
    _ = sw._real_run_training(cfg0)
    eff = captured_cfg[0]
    # Defaults preserved when augment disabled
    assert eff.augment is False
    assert eff.noise_prob == 0.0
    assert eff.dots_prob == 0.0
    assert eff.dots_count == 0
    assert eff.dots_size_px == 1


def test_space_audit_reports_top_items(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b").mkdir(parents=True, exist_ok=True)
    (root / "a" / "x.bin").write_bytes(b"0" * 100)
    (root / "b" / "y.bin").write_bytes(b"0" * 200)
    # Invoke audit helper directly
    from scripts.worker import _compute_space_audit as audit
    from scripts.worker import _format_space_audit as fmt

    files, dirs = audit(root, n_files=2, n_dirs=2)
    assert len(files) >= 2 and files[0][1] >= files[1][1]
    # Directory b should rank at least as large as a
    names = [p.name for p, _ in dirs]
    assert "b" in names
    # JSON-friendly formatting
    payload = fmt(root, files, dirs)
    assert payload.get("root") == root.as_posix()
    assert isinstance(payload.get("top_files"), list) and isinstance(payload.get("top_dirs"), list)
