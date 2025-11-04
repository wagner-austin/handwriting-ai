from __future__ import annotations

import json
from pathlib import Path

from scripts.prestart import SeedPlan, apply_seed


def _write_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _write_manifest(p: Path, **fields: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fields), encoding="utf-8")


def test_apply_seed_if_missing(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    mid = "mA"
    _write_bytes(seed_root / mid / "model.pt", b"pt")
    _write_manifest(
        seed_root / mid / "manifest.json",
        schema_version="v1",
        model_id=mid,
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at="2025-01-01T00:00:00+00:00",
        preprocess_hash="h",
        val_acc=0.9,
        temperature=1.0,
    )
    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=mid)
    act = apply_seed(plan, policy="if_missing", backup=False, backup_dir=tmp_path / "bk")
    assert act == "seeded_missing"
    assert (model_dir / mid / "model.pt").exists()


def test_apply_seed_if_newer_with_backup(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    mid = "mB"
    # existing older runtime
    _write_bytes(model_dir / mid / "model.pt", b"old")
    _write_manifest(
        model_dir / mid / "manifest.json",
        schema_version="v1",
        model_id=mid,
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at="2025-01-01T00:00:00+00:00",
        preprocess_hash="h",
        val_acc=0.9,
        temperature=1.0,
    )
    # newer seed
    _write_bytes(seed_root / mid / "model.pt", b"new")
    _write_manifest(
        seed_root / mid / "manifest.json",
        schema_version="v1",
        model_id=mid,
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at="2025-02-01T00:00:00+00:00",
        preprocess_hash="h",
        val_acc=0.95,
        temperature=1.0,
    )
    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=mid)
    backup_dir = tmp_path / "bk"
    act = apply_seed(plan, policy="if_newer", backup=True, backup_dir=backup_dir)
    assert act == "seeded_updated"
    # backup present
    bk_dirs = list(backup_dir.glob(f"{mid}*"))
    assert len(bk_dirs) >= 1
    # runtime replaced
    assert (model_dir / mid / "model.pt").read_bytes() == b"new"


def test_apply_seed_never(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    mid = "mC"
    _write_bytes(seed_root / mid / "model.pt", b"x")
    _write_manifest(
        seed_root / mid / "manifest.json",
        schema_version="v1",
        model_id=mid,
        arch="resnet18",
        n_classes=10,
        version="1",
        created_at="2025-02-01T00:00:00+00:00",
        preprocess_hash="h",
        val_acc=0.95,
        temperature=1.0,
    )
    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=mid)
    act = apply_seed(plan, policy="never", backup=True, backup_dir=tmp_path / "bk")
    assert act == "skipped_never"
