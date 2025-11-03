from __future__ import annotations

from pathlib import Path

from scripts.prestart import SeedPlan, seed_if_needed


def _write(p: Path, content: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_seed_if_needed_copies_when_missing(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    model_id = "m1"
    _write(seed_root / model_id / "model.pt", b"pt")
    _write(seed_root / model_id / "manifest.json", b"{}")

    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=model_id)
    copied = seed_if_needed(plan)
    assert copied is True
    assert (model_dir / model_id / "model.pt").exists()
    assert (model_dir / model_id / "manifest.json").exists()


def test_seed_if_needed_skips_when_present(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    model_id = "m2"
    _write(seed_root / model_id / "model.pt", b"pt")
    _write(seed_root / model_id / "manifest.json", b"{}")
    _write(model_dir / model_id / "model.pt", b"pt")
    _write(model_dir / model_id / "manifest.json", b"{}")

    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=model_id)
    copied = seed_if_needed(plan)
    assert copied is False


def test_seed_if_needed_no_seed_assets(tmp_path: Path) -> None:
    seed_root = tmp_path / "seed" / "digits" / "models"
    model_dir = tmp_path / "data" / "digits" / "models"
    model_id = "m3"
    plan = SeedPlan(seed_root=seed_root, model_dir=model_dir, model_id=model_id)
    copied = seed_if_needed(plan)
    assert copied is False
