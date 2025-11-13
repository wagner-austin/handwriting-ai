from __future__ import annotations

from pathlib import Path

import pytest
from scripts.train_mnist_resnet18 import (
    _apply_overrides,
    _default_config,
    _parse_args,
    _read_defaults,
)
from scripts.train_mnist_resnet18 import (
    main as train_main,
)


def test_default_config_and_overrides(tmp_path: Path) -> None:
    cfg = _default_config()
    assert cfg.batch_size > 0 and cfg.model_id

    # Build a config TOML with partial overrides
    toml = b"[trainer]\nmodel_id='m2'\nbatch_size=64\nlr=0.002\naugment=true\n"
    p = tmp_path / "trainer.toml"
    p.write_bytes(toml)

    cfg2 = _read_defaults(p)
    assert cfg2.model_id == "m2" and cfg2.batch_size == 64 and cfg2.lr == 0.002 and cfg2.augment

    # Apply programmatic overrides
    cfg3 = _apply_overrides(cfg2, {"epochs": 1, "aug_translate": 0.2})
    assert cfg3.epochs == 1 and cfg3.aug_translate == 0.2


def test_parse_args_and_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    # Build a fake datasets.MNIST to avoid downloads
    class _DS:
        def __init__(self, *_: object, **__: object) -> None:
            self._n = 8

        def __len__(self) -> int:
            return self._n

    # Patch torchvision.datasets and training entry

    # Patch via dotted path to avoid mypy attribute warnings
    monkeypatch.setattr("scripts.train_mnist_resnet18.datasets.MNIST", _DS, raising=True)

    called = {"n": 0}

    def _twc(cfg: object, bases: tuple[object, object]) -> None:
        called["n"] += 1

    monkeypatch.setattr("scripts.train_mnist_resnet18.train_with_config", _twc, raising=True)

    # Build a minimal config file and args
    cfg_file = tmp_path / "cfg.toml"
    cfg_file.write_text("[trainer]\nmodel_id='m3'\n", encoding="utf-8")
    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "trainer.py",
            "--config",
            str(cfg_file),
            "--epochs",
            "1",
            "--log-style",
            "pretty",
        ]
        cfg, style = _parse_args()
        assert style == "pretty" and int(cfg.epochs) == 1
        train_main()
        assert called["n"] >= 1
    finally:
        sys.argv = argv_backup
