from __future__ import annotations

from pathlib import Path

from handwriting_ai.training.artifacts import prune_model_artifacts


def _touch(p: Path, content: bytes = b"x") -> None:
    p.write_bytes(content)


def test_prune_keeps_newest_two_and_canonical(tmp_path: Path) -> None:
    d = tmp_path / "m1"
    d.mkdir(parents=True, exist_ok=True)
    # Canonical
    _touch(d / "model.pt")
    _touch(d / "manifest.json", b"{}")
    # Unique runs
    runs = [
        "20250101-000000-aaa",
        "20250101-000100-bbb",
        "20250101-000200-ccc",
        "20250101-000300-ddd",
        "20250101-000400-eee",
    ]
    for r in runs:
        _touch(d / f"model-{r}.pt")
        _touch(d / f"manifest-{r}.json", b"{}")

    deleted = prune_model_artifacts(d, keep_runs=2)
    # Oldest three should be gone
    for r in runs[:3]:
        assert not (d / f"model-{r}.pt").exists()
        assert not (d / f"manifest-{r}.json").exists()
    # Newest two remain
    for r in runs[3:]:
        assert (d / f"model-{r}.pt").exists()
        assert (d / f"manifest-{r}.json").exists()
    # Canonical preserved
    assert (d / "model.pt").exists()
    assert (d / "manifest.json").exists()
    assert len(deleted) >= 6


def test_prune_handles_strays_and_keep_zero(tmp_path: Path) -> None:
    d = tmp_path / "m2"
    d.mkdir(parents=True, exist_ok=True)
    _touch(d / "model.pt")
    _touch(d / "manifest.json", b"{}")
    # One-sided runs and a stray
    _touch(d / "model-20250101-000000-xxx.pt")
    _touch(d / "manifest-20250101-000100-yyy.json", b"{}")
    _touch(d / "stray.txt", b"u")

    deleted = prune_model_artifacts(d, keep_runs=0)
    # All unique snapshots removed
    assert not (d / "model-20250101-000000-xxx.pt").exists()
    assert not (d / "manifest-20250101-000100-yyy.json").exists()
    # Stray and canonical remain
    assert (d / "stray.txt").exists()
    assert (d / "model.pt").exists()
    assert (d / "manifest.json").exists()
    assert len(deleted) == 2


def test_prune_missing_dir_ok(tmp_path: Path) -> None:
    # Directory does not exist -> iterdir raises, function returns []
    missing = tmp_path / "nope" / "m3"
    out = prune_model_artifacts(missing, keep_runs=1)
    assert out == []


def test_prune_delete_failure_is_tolerated(tmp_path: Path) -> None:
    d = tmp_path / "m4"
    d.mkdir(parents=True, exist_ok=True)
    # Create a directory that matches unique snapshot pattern to force unlink error
    bad_dir = d / "model-20250101-000000-bad.pt"
    bad_dir.mkdir(parents=False, exist_ok=True)
    out = prune_model_artifacts(d, keep_runs=0)
    # Directory should still exist; deletion failure tolerated
    assert bad_dir.exists()
    # No files deleted by our setup
    assert isinstance(out, list)


def test_prune_no_unique_returns_empty(tmp_path: Path) -> None:
    d = tmp_path / "m5"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"pt")
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    out = prune_model_artifacts(d, keep_runs=2)
    assert out == []
