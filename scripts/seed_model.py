from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeedArgs:
    model_id: str
    from_dir: Path
    to_dir: Path


def parse_args() -> SeedArgs:
    ap = argparse.ArgumentParser(description="Copy a trained model into the seed directory")
    ap.add_argument("--model-id", required=True, help="Model id folder name")
    ap.add_argument("--from-dir", default="./artifacts/digits/models", help="Source models root")
    ap.add_argument("--to-dir", default="./seed/digits/models", help="Destination seed root")
    a = ap.parse_args()
    return SeedArgs(
        model_id=str(a.model_id),
        from_dir=Path(str(a.from_dir)),
        to_dir=Path(str(a.to_dir)),
    )


def copy_model(args: SeedArgs) -> None:
    src = args.from_dir / args.model_id
    dst = args.to_dir / args.model_id
    src_model = src / "model.pt"
    src_manifest = src / "manifest.json"
    if not (src_model.exists() and src_manifest.exists()):
        raise SystemExit(
            f"Source files not found: {src_model.as_posix()} and {src_manifest.as_posix()}"
        )
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_model, dst / "model.pt")
    shutil.copy2(src_manifest, dst / "manifest.json")
    logging.getLogger("handwriting_ai").info(
        "seed_model_copied model_id=%s src=%s dst=%s",
        args.model_id,
        src.as_posix(),
        dst.as_posix(),
    )


def main() -> None:  # pragma: no cover - tiny glue
    from handwriting_ai.logging import init_logging

    init_logging()
    args = parse_args()
    copy_model(args)


if __name__ == "__main__":
    main()
