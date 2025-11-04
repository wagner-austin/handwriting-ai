from __future__ import annotations

from datetime import UTC, datetime

from handwriting_ai.inference.manifest import ModelManifest


def test_manifest_from_json_positive() -> None:
    now = datetime.now(UTC).isoformat()
    js = (
        "{"
        + '"schema_version":"v1.1",'
        + '"model_id":"m",'
        + '"arch":"resnet18",'
        + '"n_classes":10,'
        + '"version":"1.0.0",'
        + f'"created_at":"{now}",'
        + '"preprocess_hash":"abc",'
        + '"val_acc":0.9,'
        + '"temperature":1.0'
        + "}"
    )
    man = ModelManifest.from_json(js)
    assert man.model_id == "m" and int(man.n_classes) == 10
