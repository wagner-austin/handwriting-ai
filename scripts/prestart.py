from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from handwriting_ai.config import Settings
from handwriting_ai.inference.manifest import ModelManifest
from handwriting_ai.logging import init_logging


@dataclass(frozen=True)
class SeedPlan:
    seed_root: Path
    model_dir: Path
    model_id: str


SeedPolicy = Literal["if_missing", "if_newer", "always", "never"]


def _read_manifest(path: Path) -> ModelManifest | None:
    try:
        return ModelManifest.from_path(path)
    except (OSError, ValueError):
        logging.getLogger("handwriting_ai").info(
            "prestart_manifest_read_failed",
        )
        return None


def _backup_existing(dest: Path, backup_root: Path, man: ModelManifest | None) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = ts
    out_dir = backup_root / f"{dest.name}-{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    m = dest / "model.pt"
    j = dest / "manifest.json"
    if m.exists():
        shutil.copy2(m, out_dir / "model.pt")
    if j.exists():
        shutil.copy2(j, out_dir / "manifest.json")


def apply_seed(plan: SeedPlan, *, policy: SeedPolicy, backup: bool, backup_dir: Path) -> str:
    """Apply seeding based on policy.

    Returns an action string describing what happened.
    """
    dest = plan.model_dir / plan.model_id
    seed = plan.seed_root / plan.model_id

    dest_model = dest / "model.pt"
    dest_manifest = dest / "manifest.json"
    seed_model = seed / "model.pt"
    seed_manifest = seed / "manifest.json"

    seed_exists = seed_model.exists() and seed_manifest.exists()
    dest_exists = dest_model.exists() and dest_manifest.exists()

    if policy == "never":
        return "skipped_never"
    if not seed_exists:
        return "skipped_no_seed"
    if not dest_exists:
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_model, dest_model)
        shutil.copy2(seed_manifest, dest_manifest)
        return "seeded_missing"
    if policy == "if_missing":
        return "skipped_present"

    # Compare manifests for if_newer logic
    seed_man = _read_manifest(seed_manifest)
    dest_man = _read_manifest(dest_manifest)
    if seed_man is None or dest_man is None:
        return "skipped_compare_failed"

    do_copy = policy == "always" or bool(seed_man.created_at > dest_man.created_at)
    if not do_copy:
        return "skipped_not_newer"

    if backup:
        _backup_existing(dest, backup_dir, dest_man)
    shutil.copy2(seed_model, dest_model)
    shutil.copy2(seed_manifest, dest_manifest)
    return "seeded_updated"


def seed_if_needed(plan: SeedPlan) -> bool:
    """Original behavior: only seed if missing.

    Returns True if files were copied, False otherwise.
    """
    act = apply_seed(plan, policy="if_missing", backup=False, backup_dir=Path("/tmp"))
    return act == "seeded_missing"


def _env_policy(val: str | None) -> SeedPolicy:
    if val == "always":
        return "always"
    if val == "if_newer":
        return "if_newer"
    if val == "if_missing":
        return "if_missing"
    if val == "never":
        return "never"
    return "if_missing"


def _env_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


def main() -> None:
    # Ensure logging is initialized and honors propagation settings for tests
    init_logging()
    # Load settings (env + optional TOML) to determine model_dir and active model
    settings = Settings.load()
    seed_root = Path("/seed/digits/models")
    plan = SeedPlan(
        seed_root=seed_root,
        model_dir=settings.digits.model_dir,
        model_id=settings.digits.active_model,
    )

    policy = _env_policy(os.getenv("PRESTART_SEED_POLICY"))
    backup_enabled = _env_truthy(os.getenv("PRESTART_SEED_BACKUP", "1"))
    backup_dir_env = os.getenv("PRESTART_BACKUP_DIR", "/data/backups")
    backup_dir = Path(backup_dir_env)

    action = apply_seed(plan, policy=policy, backup=backup_enabled, backup_dir=backup_dir)
    logging.getLogger("handwriting_ai").info(
        "prestart seed_policy=%s action=%s model_id=%s", policy, action, plan.model_id
    )


if __name__ == "__main__":  # pragma: no cover - used at runtime
    main()
