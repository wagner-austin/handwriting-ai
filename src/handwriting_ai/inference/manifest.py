from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    model_id: str
    arch: str
    n_classes: int
    version: str
    created_at: datetime
    preprocess_hash: str
    val_acc: float
    temperature: float

    @staticmethod
    def from_path(path: Path) -> ModelManifest:
        obj: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("manifest must be a JSON object")
        data: dict[str, object] = {str(k): v for k, v in obj.items()}
        return ModelManifest.from_dict(data)

    @staticmethod
    def from_json(s: str) -> ModelManifest:
        obj: object = json.loads(s)
        if not isinstance(obj, dict):
            raise ValueError("manifest must be a JSON object")
        data: dict[str, object] = {str(k): v for k, v in obj.items()}
        return ModelManifest.from_dict(data)

    @staticmethod
    def from_dict(d: dict[str, object]) -> ModelManifest:
        allowed_schema_versions: Final[tuple[str, ...]] = ("v1.1",)
        created_at_str = str(d["created_at"]) if "created_at" in d else ""
        created = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
        n_classes = int(str(d.get("n_classes", 10)))
        val_acc = float(str(d.get("val_acc", 0.0)))
        temperature = float(str(d.get("temperature", 1.0)))
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not (0.0 <= val_acc <= 1.0):
            raise ValueError("val_acc must be within [0,1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        schema_version = str(d.get("schema_version", "")).strip()
        model_id = str(d.get("model_id", "")).strip()
        arch = str(d.get("arch", "")).strip()
        version = str(d.get("version", "")).strip()
        preprocess_hash = str(d.get("preprocess_hash", "")).strip()
        if not schema_version or not model_id or not arch or not version or not preprocess_hash:
            raise ValueError("manifest is missing required fields")
        if schema_version not in allowed_schema_versions:
            raise ValueError("unsupported manifest schema version")
        return ModelManifest(
            schema_version=schema_version,
            model_id=model_id,
            arch=arch,
            n_classes=n_classes,
            version=version,
            created_at=created,
            preprocess_hash=preprocess_hash,
            val_acc=val_acc,
            temperature=temperature,
        )
