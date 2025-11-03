from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from handwriting_ai.config import Settings


@dataclass(frozen=True)
class SeedPlan:
    seed_root: Path
    model_dir: Path
    model_id: str


def seed_if_needed(plan: SeedPlan) -> bool:
    """Seed the artifacts volume if the active model is missing.

    Returns True if files were copied, False otherwise.
    """
    dest = plan.model_dir / plan.model_id
    seed = plan.seed_root / plan.model_id

    dest_model = dest / "model.pt"
    dest_manifest = dest / "manifest.json"
    if dest_model.exists() and dest_manifest.exists():
        return False

    seed_model = seed / "model.pt"
    seed_manifest = seed / "manifest.json"
    if not (seed_model.exists() and seed_manifest.exists()):
        # Nothing to do; seed assets not present in image
        return False

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_model, dest_model)
    shutil.copy2(seed_manifest, dest_manifest)
    return True


def main() -> None:
    # Load settings (env + optional TOML) to determine model_dir and active model
    settings = Settings.load()
    seed_root = Path("/seed/digits/models")
    plan = SeedPlan(
        seed_root=seed_root,
        model_dir=settings.digits.model_dir,
        model_id=settings.digits.active_model,
    )
    copied = seed_if_needed(plan)
    # Print simple messages; rely on platform logs for visibility
    if copied:
        print(f"seeded artifacts for model_id={plan.model_id} into {plan.model_dir}")
    else:
        print("no seeding needed or seed not present; continuing")


if __name__ == "__main__":  # pragma: no cover - used at runtime
    main()
