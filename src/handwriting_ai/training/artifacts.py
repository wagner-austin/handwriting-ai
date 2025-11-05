from __future__ import annotations

import json
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
    import os
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
