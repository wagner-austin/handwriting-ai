from __future__ import annotations

from pathlib import Path

import pytest
from scripts import seed_model as sm
from scripts.seed_model import SeedArgs, copy_model


def _write(p: Path, content: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_copy_model_success(tmp_path: Path) -> None:
    src = tmp_path / "artifacts/digits/models/mid"
    dst = tmp_path / "seed/digits/models"
    _write(src / "model.pt", b"x")
    _write(src / "manifest.json", b"{}")
    args = SeedArgs(model_id="mid", from_dir=src.parent, to_dir=dst)
    copy_model(args)
    assert (dst / "mid/model.pt").exists() and (dst / "mid/manifest.json").exists()


def test_copy_model_missing_raises(tmp_path: Path) -> None:
    src = tmp_path / "artifacts/digits/models/mid"
    dst = tmp_path / "seed/digits/models"
    args = SeedArgs(model_id="mid", from_dir=src.parent, to_dir=dst)
    with pytest.raises(SystemExit):
        copy_model(args)


def test_parse_args_and_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create a source model and run main via parse_args
    src_root = tmp_path / "artifacts/digits/models"
    dst_root = tmp_path / "seed/digits/models"
    (src_root / "m2").mkdir(parents=True, exist_ok=True)
    (src_root / "m2/model.pt").write_bytes(b"x")
    (src_root / "m2/manifest.json").write_bytes(b"{}")

    import sys

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "seed_model.py",
            "--model-id",
            "m2",
            "--from-dir",
            str(src_root),
            "--to-dir",
            str(dst_root),
        ]
        args = sm.parse_args()
        assert args.model_id == "m2" and args.from_dir == src_root and args.to_dir == dst_root
        sm.main()
        assert (dst_root / "m2/model.pt").exists()
    finally:
        sys.argv = argv_backup
