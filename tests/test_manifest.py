from __future__ import annotations

from datetime import UTC, datetime

from handwriting_ai.inference.manifest import ModelManifest


def test_manifest_from_dict_valid() -> None:
    d: dict[str, object] = {
        "schema_version": "v1",
        "model_id": "mnist_resnet18_v1",
        "arch": "resnet18_cifar",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm",
        "val_acc": 0.99,
        "temperature": 1.0,
    }
    man = ModelManifest.from_dict(d)
    assert man.model_id == "mnist_resnet18_v1"
    assert man.n_classes == 10


def test_manifest_from_dict_missing_raises() -> None:
    bad: dict[str, object] = {
        "schema_version": "v1",
        # missing model_id
        "arch": "resnet18_cifar",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "preprocess_hash": "abc",
        "val_acc": 0.98,
        "temperature": 1.0,
    }
    raised = False
    try:
        _ = ModelManifest.from_dict(bad)
    except ValueError:
        raised = True
    assert raised is True
