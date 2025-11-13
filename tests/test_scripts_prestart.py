from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts.prestart import (
    SeedPlan,
    _backup_existing,
    _env_policy,
    _env_truthy,
    _read_manifest,
    apply_seed,
    seed_if_needed,
)
from scripts.prestart import (
    main as prestart_main,
)


def _write(p: Path, content: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def _manifest(created: datetime) -> bytes:
    return (
        b'{"schema_version":"v1.1","model_id":"m","arch":"r18","n_classes":10,'
        b'"version":"0","created_at":"' + created.isoformat().encode("utf-8") + b'",'
        b'"preprocess_hash":"x","val_acc":0.9,"temperature":1.0}'
    )


def test_env_helpers() -> None:
    assert _env_policy("always") == "always"
    assert _env_policy("if_newer") == "if_newer"
    assert _env_policy("if_missing") == "if_missing"
    assert _env_policy("never") == "never"
    assert _env_policy("bogus") == "if_missing"
    assert _env_truthy("1") and _env_truthy("YES") and not _env_truthy(None)


def test_seed_if_needed_missing(tmp_path: Path) -> None:
    seed = tmp_path / "seed/digits/models"
    dest = tmp_path / "artifacts/digits/models"
    # Create seed model
    _write(seed / "m/model.pt", b"x")
    _write(seed / "m/manifest.json", _manifest(datetime.now()))
    plan = SeedPlan(seed_root=seed, model_dir=dest, model_id="m")
    assert seed_if_needed(plan) is True
    assert (dest / "m/model.pt").exists()


def test_apply_seed_policies(tmp_path: Path) -> None:
    seed = tmp_path / "seed/digits/models"
    dest = tmp_path / "artifacts/digits/models"
    older = datetime.now() - timedelta(days=1)
    newer = datetime.now()

    # Seed files
    _write(seed / "m/model.pt", b"x")
    _write(seed / "m/manifest.json", _manifest(newer))

    # Dest existing older manifest
    _write(dest / "m/model.pt", b"y")
    _write(dest / "m/manifest.json", _manifest(older))

    plan = SeedPlan(seed_root=seed, model_dir=dest, model_id="m")

    # never
    assert apply_seed(plan, policy="never", backup=False, backup_dir=tmp_path) == "skipped_never"
    # if_missing
    act_missing = apply_seed(plan, policy="if_missing", backup=False, backup_dir=tmp_path)
    assert act_missing == "skipped_present"
    # if_newer -> copies and returns seeded_updated
    act = apply_seed(plan, policy="if_newer", backup=True, backup_dir=tmp_path / "bk")
    assert act == "seeded_updated" and (dest / "m/model.pt").exists()
    # always -> copies again
    act2 = apply_seed(plan, policy="always", backup=False, backup_dir=tmp_path)
    assert act2 == "seeded_updated"


def test_read_manifest_and_backup(tmp_path: Path) -> None:
    # _read_manifest failure path
    bad = tmp_path / "bad.json"
    bad.write_text("notjson", encoding="utf-8")
    assert _read_manifest(bad) is None

    # _backup_existing copies whichever files exist
    dest = tmp_path / "dest/modelA"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model.pt").write_bytes(b"x")
    backup_dir = tmp_path / "bk"
    _backup_existing(dest, backup_dir, None)
    # Ensure backup directory created
    assert backup_dir.exists()


def test_prestart_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Stub Settings.load inside module
    class _Digits:
        def __init__(self, model_dir: Path, active_model: str) -> None:
            self.model_dir = model_dir
            self.active_model = active_model

    class _Settings:
        def __init__(self) -> None:
            self.digits = _Digits(tmp_path / "artifacts/digits/models", "demo")

        @staticmethod
        def load() -> _Settings:
            return _Settings()

    monkeypatch.setattr("scripts.prestart.Settings", _Settings, raising=True)

    # Create seed sources referenced by main
    seed_root = Path("/seed/digits/models")
    (seed_root / "demo").mkdir(parents=True, exist_ok=True)
    (seed_root / "demo/model.pt").write_bytes(b"x")
    (seed_root / "demo/manifest.json").write_text(
        '{"schema_version":"v1.1","model_id":"demo","arch":"r18","n_classes":10,"version":"0","created_at":"2025-01-01T00:00:00","preprocess_hash":"h","val_acc":0.9,"temperature":1.0}',
        encoding="utf-8",
    )

    monkeypatch.setenv("PRESTART_SEED_POLICY", "always")
    monkeypatch.setenv("PRESTART_SEED_BACKUP", "1")
    monkeypatch.setenv("PRESTART_BACKUP_DIR", str(tmp_path / "bk"))
    # Ensure structured logs propagate to root for caplog
    monkeypatch.setenv("HANDWRITING_LOG_PROPAGATE", "1")
    caplog.set_level(logging.INFO, logger="handwriting_ai")
    prestart_main()
    messages = [rec.message for rec in caplog.records if rec.name == "handwriting_ai"]
    assert any("prestart seed_policy=" in m for m in messages)
