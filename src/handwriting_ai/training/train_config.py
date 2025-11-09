from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path
    out_dir: Path
    model_id: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    optim: str
    scheduler: str
    step_size: int
    gamma: float
    min_lr: float
    patience: int
    min_delta: float
    threads: int
    augment: bool
    aug_rotate: float
    aug_translate: float
    # Optional augmentation modifiers (default-off)
    noise_prob: float = 0.0
    noise_salt_vs_pepper: float = 0.5
    dots_prob: float = 0.0
    dots_count: int = 0
    dots_size_px: int = 1
    blur_sigma: float = 0.0
    morph: str = "none"
    morph_kernel_px: int = 1
    # Progress emission cadence (epochs). 1 = every epoch.
    progress_every_epochs: int = 1
    # Progress emission cadence (batches). 0 = emit every batch.
    progress_every_batches: int = 100
    # Calibration toggles (preflight pipeline tuning)
    calibrate: bool = False
    calibration_samples: int = 8
    force_calibration: bool = False
    # Memory guard (proactive OOM prevention)
    memory_guard: bool = True
    mem_guard_threshold_pct: float = 92.0
    mem_guard_required_checks: int = 3
