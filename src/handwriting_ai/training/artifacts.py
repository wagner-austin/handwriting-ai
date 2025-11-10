from __future__ import annotations

import json
import logging
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from handwriting_ai.preprocess import preprocess_signature

MNIST_N_CLASSES: Final[int] = 10


def write_artifacts(
    *,
    out_dir: Path,
    model_id: str,
    model_state: dict[str, Tensor],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device_str: str,
    optim: str,
    scheduler: str,
    augment: bool,
    test_val_acc: float,
) -> Path:
    model_dir = out_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_rand = secrets.token_hex(3)
    run_id = f"{run_ts}-{run_rand}"
    model_unique = model_dir / f"model-{run_id}.pt"
    manifest_unique = model_dir / f"manifest-{run_id}.json"
    torch.save(model_state, model_unique.as_posix())
    # Ensure file is fully written before copying
    model_unique_size = model_unique.stat().st_size
    logging.getLogger("handwriting_ai").info(f"model_saved size_bytes={model_unique_size}")
    manifest = {
        "schema_version": "v1.1",
        "model_id": model_id,
        "arch": "resnet18",
        "n_classes": MNIST_N_CLASSES,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": preprocess_signature(),
        "val_acc": float(test_val_acc),
        "temperature": 1.0,
        # Run metadata
        "run_id": run_id,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(device_str),
        "optim": str(optim),
        "scheduler": str(scheduler),
        "augment": bool(augment),
    }
    manifest_unique.write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copy2(model_unique, model_dir / "model.pt")
    shutil.copy2(manifest_unique, model_dir / "manifest.json")
    # Log final artifact size for upload verification
    final_model_size = (model_dir / "model.pt").stat().st_size
    logging.getLogger("handwriting_ai").info(f"model_copied size_bytes={final_model_size}")
    return model_dir


def _run_id_from_name(name: str) -> tuple[str, str] | None:
    """Return (kind, run_id) where kind is 'model' or 'manifest' for unique snapshot files.

    - model-<run_id>.pt -> ("model", <run_id>)
    - manifest-<run_id>.json -> ("manifest", <run_id>)
    Returns None for canonical files and non-matching names.
    """
    if name.startswith("model-") and name.endswith(".pt"):
        rid = name[len("model-") : -len(".pt")]
        return ("model", rid) if rid else None
    if name.startswith("manifest-") and name.endswith(".json"):
        rid = name[len("manifest-") : -len(".json")]
        return ("manifest", rid) if rid else None
    return None


def _delete_paths(paths: list[Path]) -> list[Path]:
    deleted: list[Path] = []
    for p in paths:
        try:
            p.unlink()
            deleted.append(p)
        except OSError as exc:
            logging.getLogger("handwriting_ai").error(
                "prune_delete_failed path=%s error=%s", p, exc
            )
            raise
    return deleted


def prune_model_artifacts(model_dir: Path, keep_runs: int) -> list[Path]:
    """Remove older unique snapshot files, keeping the newest N runs.

    Preserves canonical files (model.pt, manifest.json).
    Keeps the newest `keep_runs` of unique snapshots identified by run_id in filenames.
    Returns a list of deleted file paths.
    """
    keep = max(0, int(keep_runs))
    try:
        entries = list(model_dir.iterdir())
    except OSError as exc:
        logging.getLogger("handwriting_ai").error(
            "prune_list_failed dir=%s error=%s", model_dir, exc
        )
        raise

    # Collect all run_ids present in either side
    run_ids = {m[1] for p in entries if (m := _run_id_from_name(p.name)) is not None}

    if keep <= 0:
        to_delete = [p for p in entries if _run_id_from_name(p.name) is not None]
        return _delete_paths(to_delete)

    if not run_ids:
        return []

    # Sort run ids lexicographically (timestamp-first names sort correctly)
    sorted_ids = sorted(run_ids)
    keep_set = set(sorted_ids[-keep:])
    to_delete = [
        p for p in entries if (m := _run_id_from_name(p.name)) is not None and m[1] not in keep_set
    ]
    return _delete_paths(to_delete)
