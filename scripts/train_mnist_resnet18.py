from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Final

from torchvision import datasets

from handwriting_ai.logging import get_logger, init_logging
from handwriting_ai.training import TrainConfig, train_with_config

MNIST_N_CLASSES: Final[int] = 10


def _default_config() -> TrainConfig:
    return TrainConfig(
        data_root=Path("./data/mnist"),
        out_dir=Path("./artifacts/digits/models"),
        model_id="mnist_resnet18_v1",
        epochs=4,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-2,
        seed=42,
        device="cpu",
        optim="adamw",
        scheduler="cosine",
        step_size=10,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=10.0,
        aug_translate=0.1,
    )


def _apply_overrides(cfg: TrainConfig, t: dict[str, object]) -> TrainConfig:
    v = t.get("data_root")
    data_root = Path(v) if isinstance(v, str) else cfg.data_root
    v = t.get("out_dir")
    out_dir = Path(v) if isinstance(v, str) else cfg.out_dir
    v = t.get("model_id")
    model_id = v if isinstance(v, str) else cfg.model_id
    v = t.get("epochs")
    epochs = v if isinstance(v, int) else cfg.epochs
    v = t.get("batch_size")
    batch_size = v if isinstance(v, int) else cfg.batch_size
    v = t.get("lr")
    lr = float(v) if isinstance(v, int | float) else cfg.lr
    v = t.get("weight_decay")
    weight_decay = float(v) if isinstance(v, int | float) else cfg.weight_decay
    v = t.get("seed")
    seed = v if isinstance(v, int) else cfg.seed
    v = t.get("device")
    device = v if isinstance(v, str) else cfg.device
    v = t.get("optim")
    optim = v if isinstance(v, str) else cfg.optim
    v = t.get("scheduler")
    scheduler = v if isinstance(v, str) else cfg.scheduler
    v = t.get("step_size")
    step_size = v if isinstance(v, int) else cfg.step_size
    v = t.get("gamma")
    gamma = float(v) if isinstance(v, int | float) else cfg.gamma
    v = t.get("min_lr")
    min_lr = float(v) if isinstance(v, int | float) else cfg.min_lr
    v = t.get("patience")
    patience = v if isinstance(v, int) else cfg.patience
    v = t.get("min_delta")
    min_delta = float(v) if isinstance(v, int | float) else cfg.min_delta
    v = t.get("threads")
    threads = v if isinstance(v, int) else cfg.threads
    v = t.get("augment")
    augment = v if isinstance(v, bool) else cfg.augment
    v = t.get("aug_rotate")
    aug_rotate = float(v) if isinstance(v, int | float) else cfg.aug_rotate
    v = t.get("aug_translate")
    aug_translate = float(v) if isinstance(v, int | float) else cfg.aug_translate

    return TrainConfig(
        data_root=data_root,
        out_dir=out_dir,
        model_id=model_id,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
        optim=optim,
        scheduler=scheduler,
        step_size=step_size,
        gamma=gamma,
        min_lr=min_lr,
        patience=patience,
        min_delta=min_delta,
        threads=threads,
        augment=augment,
        aug_rotate=aug_rotate,
        aug_translate=aug_translate,
    )


def _read_defaults(path: Path | None) -> TrainConfig:
    cfg = _default_config()
    if path is None or not path.exists():
        return cfg
    with path.open("rb") as f:
        parsed = tomllib.load(f)
    t = parsed.get("trainer", {})
    return _apply_overrides(cfg, t if isinstance(t, dict) else {})


def _parse_args() -> TrainConfig:
    # Stage 1: read --config to use as defaults
    ap0 = argparse.ArgumentParser(add_help=False)
    ap0.add_argument("--config", type=str, default="./config/trainer.toml")
    ns0, _ = ap0.parse_known_args()
    cfg_path = Path(ns0.config) if ns0.config else None
    defaults = _read_defaults(cfg_path)

    # Stage 2: full parser with defaults from config
    ap = argparse.ArgumentParser(description="Train MNIST ResNet-18 with service preprocess")
    ap.add_argument("--config", type=str, default=str(cfg_path) if cfg_path else None)
    ap.add_argument("--data-root", default=str(defaults.data_root))
    ap.add_argument("--out-dir", default=str(defaults.out_dir))
    ap.add_argument("--model-id", default=defaults.model_id)
    ap.add_argument("--epochs", type=int, default=defaults.epochs)
    ap.add_argument("--batch-size", type=int, default=defaults.batch_size)
    ap.add_argument("--lr", type=float, default=defaults.lr)
    ap.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=defaults.device)
    ap.add_argument("--optim", choices=["sgd", "adam", "adamw"], default=defaults.optim)
    ap.add_argument("--scheduler", choices=["none", "cosine", "step"], default=defaults.scheduler)
    ap.add_argument("--step-size", type=int, default=defaults.step_size)
    ap.add_argument("--gamma", type=float, default=defaults.gamma)
    ap.add_argument("--min-lr", type=float, default=defaults.min_lr)
    ap.add_argument("--patience", type=int, default=defaults.patience)
    ap.add_argument("--min-delta", type=float, default=defaults.min_delta)
    ap.add_argument("--threads", type=int, default=defaults.threads)
    ap.add_argument("--augment", action="store_true", default=defaults.augment)
    ap.add_argument("--aug-rotate", type=float, default=defaults.aug_rotate)
    ap.add_argument("--aug-translate", type=float, default=defaults.aug_translate)
    args = ap.parse_args()
    return TrainConfig(
        data_root=Path(str(args.data_root)),
        out_dir=Path(str(args.out_dir)),
        model_id=str(args.model_id),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        device=str(args.device),
        optim=str(args.optim),
        scheduler=str(args.scheduler),
        step_size=int(args.step_size),
        gamma=float(args.gamma),
        min_lr=float(args.min_lr),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        threads=int(args.threads),
        augment=bool(args.augment),
        aug_rotate=float(args.aug_rotate),
        aug_translate=float(args.aug_translate),
    )


def main() -> None:
    init_logging()
    log = get_logger()
    cfg = _parse_args()
    log.info(
        f"trainer_cli_start data_root={cfg.data_root} out_dir={cfg.out_dir} model_id={cfg.model_id}"
    )
    log.info("loading_mnist_train")
    train_base = datasets.MNIST(cfg.data_root.as_posix(), train=True, download=True)
    log.info("loading_mnist_test")
    test_base = datasets.MNIST(cfg.data_root.as_posix(), train=False, download=True)
    log.info(
        f"mnist_loaded train={len(train_base)} test={len(test_base)} batch_size={cfg.batch_size}"
    )
    # Calibration mode: run 1 epoch automatically to time and recommend epochs
    if int(cfg.epochs) == 1:
        import time as _time

        t0 = _time.perf_counter()
        train_with_config(cfg, (train_base, test_base))
        dt = _time.perf_counter() - t0
        hours = dt / 3600.0
        target_hours = 6.0
        # Round down to be safe
        rec_epochs = int(target_hours / hours) if hours > 0 else 0
        log.info(
            f"calibration_epoch_time_s={dt:.1f} target_hours={target_hours:.1f} "
            f"recommended_epochs={rec_epochs}"
        )
    else:
        train_with_config(cfg, (train_base, test_base))


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
