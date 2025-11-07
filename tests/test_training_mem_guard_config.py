from __future__ import annotations

from pathlib import Path

import handwriting_ai.training.mnist_train as mt
from handwriting_ai.training.resources import ResourceLimits
from handwriting_ai.training.safety import get_memory_guard_config


def _cfg() -> mt.TrainConfig:
    return mt.TrainConfig(
        data_root=Path("."),
        out_dir=Path("."),
        model_id="m",
        epochs=1,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-2,
        seed=0,
        device="cpu",
        optim="adamw",
        scheduler="none",
        step_size=1,
        gamma=0.5,
        min_lr=1e-5,
        patience=0,
        min_delta=5e-4,
        threads=0,
        augment=False,
        aug_rotate=0.0,
        aug_translate=0.0,
    )


def test_mem_guard_config_small_memory() -> None:
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=1 * 1024 * 1024 * 1024,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur.enabled is True and cur.required_consecutive >= 3 and cur.threshold_percent <= 95.0


def test_mem_guard_config_unknown_memory() -> None:
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=2,
        memory_bytes=None,
        optimal_threads=1,
        optimal_workers=0,
        max_batch_size=64,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    assert cur.enabled is bool(cfg.memory_guard)


def test_mem_guard_config_large_memory() -> None:
    cfg = _cfg()
    limits = ResourceLimits(
        cpu_cores=4,
        memory_bytes=8 * 1024 * 1024 * 1024,
        optimal_threads=4,
        optimal_workers=2,
        max_batch_size=512,
    )
    mt._configure_memory_guard_from_limits(cfg, limits)
    cur = get_memory_guard_config()
    # Should respect config defaults when memory is ample
    assert cur.enabled is True and cur.required_consecutive == cfg.mem_guard_required_checks
