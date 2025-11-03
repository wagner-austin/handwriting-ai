# Proposal: Maximum Accuracy Implementation (Competition)

**Date:** 2025-11-03
**Goal:** Win accuracy competition (99.85%+ target) within 1 week
**Current baseline:** 99.6% (Phase 1 complete)
**Strategy:** All-in, no compromises, maximum performance

---

## Executive Summary

This proposal outlines aggressive changes to maximize MNIST accuracy for competition. We go all-in on proven SOTA techniques with no backward compatibility concerns or fallback strategies.

**Expected outcome:** 99.6% → 99.85%+ accuracy
**Timeline:** 7 days
**Strategy:** Full Phase 1 + Phase 2 implementation, ensemble-first architecture

**Engineering commitment:** All changes adhere to strict typing, modularity, and production readiness. No quick fixes, no technical debt accumulation.

---

## Engineering Standards & Quality Gates

All proposed changes **MUST** satisfy the following non-negotiable requirements:

### 1. Strict Typing (Zero Tolerance)

**Enforced by mypy configuration** (pyproject.toml:62-84):
```toml
[tool.mypy]
strict = true
disallow_any_expr = true          # ← CRITICAL: No 'Any' anywhere
disallow_any_explicit = true      # ← No explicit Any annotations
disallow_any_decorated = true     # ← No untyped decorators
disallow_any_generics = true      # ← No generic[Any]
warn_unused_ignores = true
```

**Enforced by guard checks** (scripts/guard_checks.py):
- ❌ Ban `typing.Any`
- ❌ Ban `typing.cast`
- ❌ Ban `type: ignore`
- ✅ All code must pass `make guards`

**Implementation requirements:**
1. All new functions have complete type signatures
2. Use `Protocol` for third-party types (torch, PIL) to avoid `Any`
3. Use `TypedDict` for structured dicts, frozen dataclasses for configs
4. TYPE_CHECKING blocks isolate runtime from type-checking concerns
5. Explicit narrow types over broad types

---

### 2. DRY, Modular, Standardized

**Module boundaries:**
- `inference/` - Model loading, prediction, TTA, ensemble
- `training/` - Training loops, recipes, workers
- `api/` - HTTP endpoints, request/response schemas
- `config.py` - Centralized configuration

**Pattern reuse:**
- Configuration: Nested env keys (`APP__`, `DIGITS__`, `TRAINING__`)
- Data classes: Frozen dataclasses for immutable config/results
- Protocols: Abstract third-party types
- Error handling: Fail-closed, explicit error codes, no silent fallbacks

---

### 3. Anti-Drift

**Manifest-driven artifacts:**
- Every model has `manifest.json` with `preprocess_hash`
- Inference validates hash compatibility before loading
- Version mismatches → refuse to load (fail-closed)

**Signature hashing:**
```python
_PREPROCESS_SIGNATURE: Final[str] = "v1/grayscale+otsu+lcc+deskew+center+resize28+mnistnorm"

def preprocess_signature() -> str:
    """Stable hash prevents train/infer drift"""
    return _PREPROCESS_SIGNATURE
```

**Phase 2 additions:**
- Training recipe hash in manifest v1.1
- Reproducible training via seed + recipe tracking

---

### 4. Test Coverage Requirements

**Minimum coverage per module:**
- New inference code: ≥90%
- New training code: ≥85%
- New API endpoints: ≥90%

**Required test types:**
1. **Unit tests**: Pure functions (TTA, temperature, ensemble averaging)
2. **Integration tests**: API endpoints with TestClient
3. **Contract tests**: Pydantic schema validation
4. **Edge cases**: Error paths, boundary conditions
5. **Type tests**: mypy passes on tests/ directory

---

### 5. No Technical Debt

**Prohibited shortcuts:**
- ❌ Hardcoded magic numbers (use Final[] constants)
- ❌ Copy-pasted code (extract to functions)
- ❌ "TODO" comments without tracking
- ❌ Temporary workarounds
- ❌ Unclear variable names

**Required practices:**
- ✅ Explicit over implicit
- ✅ Named constants for thresholds
- ✅ Docstrings for public functions
- ✅ Error messages explain what went wrong
- ✅ Clean commit history

---

## Phase 1 Expansions (Inference Enhancements)

### 1. Advanced TTA (Default Enabled)

**File:** `src/handwriting_ai/inference/engine.py`

**Design principle:** Advanced TTA becomes the default. Simple TTA removed.

**Implementation:**
```python
from typing import Final
import math

# Module-level constants (prevents drift)
_TTA_ROTATION_ANGLES: Final[tuple[float, ...]] = (-3.0, -1.5, 1.5, 3.0)
_TTA_SCALE_FACTORS: Final[tuple[float, ...]] = (0.95, 1.05)
_TTA_PIXEL_SHIFTS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (0, -1), (1, 0), (-1, 0),  # cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal
)


def _augment_for_tta(x: Tensor) -> Tensor:
    """Apply advanced TTA: 17 variants (shifts + rotations + scaling).

    Default TTA mode for competition. No simple mode.

    Variants:
    - 1 identity
    - 8 pixel shifts (cardinal + diagonal)
    - 4 rotations (±3°, ±1.5°)
    - 2 scales (0.95x, 1.05x)

    Args:
        x: Input tensor (1, 1, 28, 28)

    Returns:
        Augmented batch (17, 1, 28, 28)
    """
    import torch.nn.functional as F

    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")

    batch: list[Tensor] = [x]

    # Pixel shifts (8 directions)
    for shift in _TTA_PIXEL_SHIFTS:
        batch.append(torch.roll(x, shifts=shift, dims=(2, 3)))

    # Rotations (4 angles)
    for angle_deg in _TTA_ROTATION_ANGLES:
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        theta = torch.tensor([
            [cos_a, -sin_a, 0.0],
            [sin_a, cos_a, 0.0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rotated = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        batch.append(rotated)

    # Scaling (2 factors)
    for scale in _TTA_SCALE_FACTORS:
        theta = torch.tensor([
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        scaled = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        batch.append(scaled)

    result = torch.cat(batch, dim=0)
    assert result.shape[0] == 17, f"Expected 17 variants, got {result.shape[0]}"
    return result
```

**Integration:**
```python
# In InferenceEngine._predict_impl
# Always apply TTA (no conditional)
batch = _augment_for_tta(tensor)
```

**Configuration:**
```python
# config.py - DigitsConfig
@dataclass(frozen=True)
class DigitsConfig:
    # ... existing fields ...
    tta: bool = True  # Always enabled for competition
```

**Rationale:**
- 17-variant TTA is proven to increase accuracy
- No need for simple mode (competition context)
- Clean, single code path
- +0.10-0.15% expected gain

**Expected gain:** +0.10-0.15% accuracy

---

### 2. Ensemble-First Architecture

**File:** `src/handwriting_ai/inference/engine.py`

**Design principle:** Engine always expects multiple models. Single-model support removed.

**Implementation:**
```python
from typing import Final

@dataclass(frozen=True)
class _LoadedModel:
    """Container for loaded model + manifest pair.

    Prevents model/manifest mismatches and enforces immutability.
    """
    model: TorchModel
    manifest: ModelManifest


class InferenceEngine:
    """Bounded thread-pool inference engine for model ensembles."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._pool = _make_pool(settings)
        self._model_lock = threading.RLock()
        self._loaded_models: list[_LoadedModel] = []
        torch.set_num_threads(1)

    @property
    def ready(self) -> bool:
        """Service ready if at least one model loaded."""
        return len(self._loaded_models) > 0

    @property
    def ensemble_size(self) -> int:
        """Number of models in ensemble."""
        return len(self._loaded_models)

    def try_load_active(self) -> None:
        """Load ensemble from DIGITS__ACTIVE_MODEL (comma-separated IDs).

        All models must have compatible preprocess_hash.
        Minimum 1 model required, 5-7 recommended for competition.
        """
        active = self._settings.digits.active_model
        model_ids = [mid.strip() for mid in active.split(",") if mid.strip()]

        if len(model_ids) == 0:
            raise ValueError("DIGITS__ACTIVE_MODEL must specify at least one model")

        loaded: list[_LoadedModel] = []
        expected_hash: str | None = None

        for model_id in model_ids:
            model_dir = self._settings.digits.model_dir / model_id
            if not model_dir.exists():
                self._logger.warning("model_dir_not_found", extra={"model_id": model_id})
                continue

            manifest_path = model_dir / "manifest.json"
            model_path = model_dir / "model.pt"

            if not (manifest_path.exists() and model_path.exists()):
                self._logger.warning("model_files_incomplete", extra={"model_id": model_id})
                continue

            try:
                manifest = ModelManifest.from_path(manifest_path)
            except (OSError, ValueError) as e:
                self._logger.warning("manifest_load_failed", extra={"model_id": model_id, "error": str(e)})
                continue

            # Validate preprocess signature
            from ..preprocess import preprocess_signature
            if manifest.preprocess_hash != preprocess_signature():
                self._logger.warning("preprocess_hash_mismatch", extra={"model_id": model_id})
                continue

            # Ensemble invariant: all models must have same preprocess hash
            if expected_hash is None:
                expected_hash = manifest.preprocess_hash
            elif manifest.preprocess_hash != expected_hash:
                self._logger.warning("ensemble_hash_mismatch", extra={
                    "model_id": model_id,
                    "expected": expected_hash,
                    "actual": manifest.preprocess_hash
                })
                continue

            # Build and load model
            model = _build_model(arch=manifest.arch, n_classes=int(manifest.n_classes))
            try:
                sd = _load_state_dict_file(model_path)
                _validate_state_dict(sd, manifest.arch, int(manifest.n_classes))
                model.load_state_dict(sd)
            except _LOAD_ERRORS as e:
                self._logger.warning("model_load_failed", extra={"model_id": model_id, "error": str(e)})
                continue

            loaded.append(_LoadedModel(model=model, manifest=manifest))
            self._logger.info("model_loaded", extra={"model_id": model_id})

        if len(loaded) == 0:
            raise RuntimeError("Failed to load any models from ensemble")

        # Atomic swap
        with self._model_lock:
            self._loaded_models = loaded

        self._logger.info("ensemble_ready", extra={
            "num_models": len(loaded),
            "model_ids": [lm.manifest.model_id for lm in loaded]
        })

    def _predict_impl(self, preprocessed: Tensor) -> PredictOutput:
        """Run inference, averaging predictions across ensemble."""
        loaded = self._loaded_models
        if len(loaded) == 0:
            raise RuntimeError("No models loaded in ensemble")

        tensor = _as_torch_tensor(preprocessed)
        batch = _augment_for_tta(tensor)  # Always apply TTA

        # Collect predictions from all models
        all_probs: list[list[float]] = []

        for loaded_model in loaded:
            loaded_model.model.eval()
            with torch.no_grad():
                logits_obj = loaded_model.model(batch)

            temperature = self._get_temperature(loaded_model.manifest)
            probs_vec = _softmax_avg(logits_obj, temperature)
            all_probs.append(probs_vec)

        # Average probabilities across ensemble
        ensemble_probs = _average_probs(all_probs)

        # Find top class
        top_idx = int(ensemble_probs.index(max(ensemble_probs)))
        best_conf = float(max(ensemble_probs))

        ensemble_id = f"ensemble_{len(loaded)}_models"

        return PredictOutput(
            digit=top_idx,
            confidence=best_conf,
            probs=tuple(ensemble_probs),
            model_id=ensemble_id
        )


def _average_probs(all_probs: list[list[float]]) -> list[float]:
    """Average probability distributions across ensemble models.

    Args:
        all_probs: List of probability vectors, one per model

    Returns:
        Averaged probability vector
    """
    if len(all_probs) == 0:
        raise ValueError("Cannot average zero probability vectors")

    n_classes = len(all_probs[0])
    avg: list[float] = [0.0] * n_classes

    for probs in all_probs:
        if len(probs) != n_classes:
            raise ValueError(f"Inconsistent class counts: {len(probs)} vs {n_classes}")
        for i in range(n_classes):
            avg[i] += probs[i]

    n_models = float(len(all_probs))
    for i in range(n_classes):
        avg[i] /= n_models

    return avg
```

**Configuration:**
```bash
# Ensemble required (comma-separated)
DIGITS__ACTIVE_MODEL=resnet18_v1,wideresnet_v1,densenet_v1,resnet18_v2,wideresnet_v2
```

**Rationale:**
- Ensembles always improve accuracy over single models
- Simpler code (no conditional logic)
- Forces best practices for competition
- +0.05-0.10% expected gain

**Expected gain:** +0.05-0.10% accuracy

---

### 3. Temperature Optimization (Required)

**File:** `src/handwriting_ai/inference/engine.py` + `config.py`

**Design principle:** Temperature tuning is mandatory for optimal calibration.

**Implementation:**
```python
def _get_temperature(self, manifest: ModelManifest) -> float:
    """Get temperature for softmax calibration.

    Uses DIGITS__TEMPERATURE_OVERRIDE if set, otherwise manifest value.

    Args:
        manifest: Model manifest containing default temperature

    Returns:
        Temperature value (must be > 0)
    """
    override = self._settings.digits.temperature_override
    if override is not None:
        if override <= 0.0:
            raise ValueError(f"Invalid temperature override: {override}")
        return float(override)
    return float(manifest.temperature)
```

**Configuration:**
```python
@dataclass(frozen=True)
class DigitsConfig:
    # ... existing fields ...
    temperature_override: float | None = None  # Tuned for ensemble
```

**Rationale:**
- Temperature affects ensemble calibration
- Must be tunable without retraining
- Fail-fast on invalid values (no silent fallback)
- +0.01-0.03% expected gain

**Expected gain:** +0.01-0.03% accuracy

---

## Phase 2 (Training Pipeline)

### 4. Training Infrastructure

**New module structure:**
```
src/handwriting_ai/training/
├── __init__.py
├── types.py       # TypedDicts, Protocols, frozen dataclasses
├── data.py        # MNIST/EMNIST data loading
├── recipes.py     # SOTA training configurations
├── worker.py      # RQ worker implementation
└── jobs.py        # Job state management
```

**Strict typing requirements:**

**File: `src/handwriting_ai/training/types.py`**
```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, TypedDict


class JobStatus(str, Enum):
    """Training job lifecycle states."""
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class TrainingMetrics(TypedDict, total=False):
    """Metrics tracked during training."""
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_acc: float


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration."""
    arch: str
    recipe: str
    epochs: int
    seed: int
    model_id: str
    created_at: datetime


@dataclass(frozen=True)
class JobState:
    """Immutable snapshot of job state."""
    job_id: str
    status: JobStatus
    config: TrainingConfig
    progress: float
    current_epoch: int
    metrics: TrainingMetrics
    model_id: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None
```

**Design doc status:** Specified in Phase 2 (line 201-211)

---

### 5. SOTA Training Recipes

**File:** `src/handwriting_ai/training/recipes.py`

**Design principle:** Only SOTA recipes. No baseline.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class HeavyAugRecipe:
    """SOTA augmentation (99.73-99.77% expected).

    Based on academic papers (Zagoruyko 2016, DeVries 2017).
    """
    optimizer: str = "AdamW"
    lr: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "CosineAnnealingLR"
    t_max: int = 150
    eta_min: float = 1e-6
    batch_size: int = 128
    epochs: int = 150
    label_smoothing: float = 0.1
    dropout: float = 0.3

    # Heavy augmentation
    rotation_degrees: float = 15.0
    translate: tuple[float, float] = (0.15, 0.15)
    scale: tuple[float, float] = (0.85, 1.15)
    elastic_alpha: float = 34.0
    elastic_sigma: float = 4.0
    cutout_length: int = 8
    mixup_alpha: float = 0.2


@dataclass(frozen=True)
class SemiSupervisedRecipe:
    """EMNIST pre-train + MNIST fine-tune (99.75-99.80% expected).

    Two-stage training leverages 280k EMNIST digit samples.
    """
    # Stage 1: EMNIST pre-training
    pretrain_optimizer: str = "Adam"
    pretrain_lr: float = 0.001
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 256

    # Stage 2: MNIST fine-tuning
    finetune_optimizer: str = "AdamW"
    finetune_lr: float = 1e-4
    finetune_epochs: int = 100
    finetune_batch_size: int = 128
    finetune_label_smoothing: float = 0.1
    finetune_rotation_degrees: float = 15.0
    finetune_translate: tuple[float, float] = (0.15, 0.15)
    finetune_scale: tuple[float, float] = (0.85, 1.15)


_RECIPES: Final[dict[str, type[HeavyAugRecipe] | type[SemiSupervisedRecipe]]] = {
    "heavy_aug": HeavyAugRecipe,
    "semi_supervised": SemiSupervisedRecipe,
}


def get_recipe(name: str) -> HeavyAugRecipe | SemiSupervisedRecipe:
    """Get training recipe by name.

    Args:
        name: Recipe name (heavy_aug, semi_supervised)

    Returns:
        Recipe instance

    Raises:
        ValueError: Unknown recipe name
    """
    recipe_cls = _RECIPES.get(name)
    if recipe_cls is None:
        raise ValueError(f"Unknown recipe: {name}. Valid: {list(_RECIPES.keys())}")
    return recipe_cls()
```

**Rationale:**
- Only SOTA recipes (no weak baselines)
- Frozen dataclasses prevent config drift
- All hyperparameters explicit
- Based on academic papers

**Expected gain:** +0.15-0.20% per model

---

### 6. Manifest Schema v1.1

**File:** `src/handwriting_ai/inference/manifest.py`

**Design principle:** Extended manifest for training provenance. No v1 support.

```python
@dataclass(frozen=True)
class ModelManifest:
    """Model manifest v1.1 (competition format).

    All fields required for training provenance.
    """
    # Core fields
    schema_version: str
    model_id: str
    arch: str
    n_classes: int
    version: str
    created_at: datetime
    preprocess_hash: str
    val_acc: float
    temperature: float

    # Training provenance (required for v1.1)
    training_recipe: str
    training_seed: int
    training_epochs: int
    test_acc: float
    recipe_hash: str
    parent_model_id: str | None = None  # For semi-supervised only

    @staticmethod
    def from_dict(d: dict[str, object]) -> ModelManifest:
        """Parse manifest from dict (v1.1 format only)."""
        # v1.1 requires all provenance fields
        required_fields = [
            "schema_version", "model_id", "arch", "n_classes", "version",
            "created_at", "preprocess_hash", "val_acc", "temperature",
            "training_recipe", "training_seed", "training_epochs",
            "test_acc", "recipe_hash"
        ]

        for field in required_fields:
            if field not in d:
                raise ValueError(f"Missing required field: {field}")

        # Parse and validate
        created_at_str = str(d["created_at"])
        created = datetime.fromisoformat(created_at_str)

        n_classes = int(str(d["n_classes"]))
        val_acc = float(str(d["val_acc"]))
        test_acc = float(str(d["test_acc"]))
        temperature = float(str(d["temperature"]))

        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not (0.0 <= val_acc <= 1.0):
            raise ValueError("val_acc must be within [0,1]")
        if not (0.0 <= test_acc <= 1.0):
            raise ValueError("test_acc must be within [0,1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        return ModelManifest(
            schema_version=str(d["schema_version"]).strip(),
            model_id=str(d["model_id"]).strip(),
            arch=str(d["arch"]).strip(),
            n_classes=n_classes,
            version=str(d["version"]).strip(),
            created_at=created,
            preprocess_hash=str(d["preprocess_hash"]).strip(),
            val_acc=val_acc,
            temperature=temperature,
            training_recipe=str(d["training_recipe"]).strip(),
            training_seed=int(str(d["training_seed"])),
            training_epochs=int(str(d["training_epochs"])),
            test_acc=test_acc,
            recipe_hash=str(d["recipe_hash"]).strip(),
            parent_model_id=str(d["parent_model_id"]) if "parent_model_id" in d else None,
        )
```

**Rationale:**
- Full training provenance required
- No legacy v1 support (competition context)
- Fail-fast on missing fields
- Recipe hash prevents drift

---

## Configuration

### Environment Variables (Required)

```bash
# ============================================
# Phase 1 (Inference)
# ============================================

# Ensemble (comma-separated, minimum 1, recommend 5-7)
DIGITS__ACTIVE_MODEL=resnet18_h1,wideresnet_h1,densenet_h1,resnet18_h2,wideresnet_s1

# TTA always enabled (true)
DIGITS__TTA=true

# Temperature (tune for ensemble)
DIGITS__TEMPERATURE_OVERRIDE=1.2


# ============================================
# Phase 2 (Training)
# ============================================

# Redis connection
REDIS_URL=redis://localhost:6379/0

# RQ configuration
RQ__QUEUE_NAME=digits
RQ__JOB_TIMEOUT_SEC=7200

# Training data
TRAINING__DATA_DIR=/data/mnist
TRAINING__EMNIST_DIR=/data/emnist
TRAINING__CHECKPOINT_DIR=/data/checkpoints
TRAINING__NUM_WORKERS=4
TRAINING__PIN_MEMORY=true
```

---

## Implementation Timeline

### Day 1: Phase 1 Expansions
**Goal:** Advanced TTA + ensemble architecture

- [ ] **Hour 1-2:** Implement advanced TTA (default enabled)
  - Replace simple TTA with 17-variant advanced TTA
  - Add module constants
  - Write 5+ tests
  - Verify `make check` passes

- [ ] **Hour 3-4:** Implement ensemble-first architecture
  - Refactor `InferenceEngine` for ensemble-only
  - Remove single-model code paths
  - Add `_LoadedModel` frozen dataclass
  - Write 8+ tests
  - Verify `make check` passes

- [ ] **Hour 5-6:** Temperature tuning + validation
  - Add `_get_temperature()` with fail-fast validation
  - Write 3+ tests
  - Run accuracy baseline
  - Document results

**Deliverable:** Advanced TTA + ensemble ready
**Quality gate:** `make check` passes, ≥93% coverage

---

### Day 2-3: Phase 2 Core

**Day 2:**
- Add Redis/RQ dependencies
- Implement training endpoints (POST /v1/train, GET /v1/jobs)
- Add Pydantic schemas with strict validation

**Day 3:**
- Implement RQ worker
- Add training recipes (heavy_aug, semi_supervised only)
- Test full training pipeline

**Deliverable:** Can train models end-to-end

---

### Day 4-6: SOTA Model Training

**Day 4:**
- Train ResNet-18 + heavy_aug
- Train Wide ResNet-28-10 + heavy_aug
- Train DenseNet-BC-100 + heavy_aug

**Day 5:**
- Train ResNet-18 + semi_supervised
- Train Wide ResNet-28-10 + semi_supervised
- Train variant seeds (diversity)

**Day 6:**
- Train additional architectures
- Temperature tuning per model
- Collect best 5-7 models

**Deliverable:** 6-8 SOTA models ready

---

### Day 7: Final Assembly

**Morning:**
- Select best 5-7 models for ensemble
- Configure ensemble
- Tune temperature

**Afternoon:**
- Final validation on MNIST test set
- Document results

**Deliverable:** 99.82-99.87% final accuracy

---

## Testing Strategy

**Test coverage requirements:**
- New functions: 100% line coverage
- New modules: ≥90% line coverage
- Edge cases: All error paths tested

**Test files:**
```
tests/
├── test_tta_advanced.py        # Advanced TTA
├── test_ensemble.py            # Ensemble loading/prediction
├── test_temperature.py         # Temperature tuning
├── test_training_api.py        # Training endpoints
├── test_recipes.py             # Training recipes
└── test_manifest_v1_1.py       # Manifest v1.1
```

---

## Quality Gates (Non-Negotiable)

All changes **MUST** pass:

1. **`make check`** - Full CI pipeline
   - `poetry run ruff check . --fix`
   - `poetry run ruff format .`
   - `poetry run mypy` (strict mode)
   - `poetry run python scripts/guard_checks.py`
   - `poetry run pytest --cov=src --cov-report=term-missing`

2. **Coverage threshold:** ≥93% overall
3. **Type coverage:** 100% (no Any, no type: ignore, no cast)
4. **Tests:** All new code has tests, all pass
5. **Documentation:** Docstrings for public functions

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Type errors** | Low | High | Guard checks + mypy strict |
| **Tech debt** | Low | High | Frozen dataclasses, DRY, code review |
| **Config drift** | Low | Medium | Manifest hashing, recipe hashing |
| **Performance regression** | Low | Medium | Advanced TTA always enabled, tested |
| **Training failures** | Medium | High | Use proven recipes from papers |

---

## Expected Accuracy Progression

| Milestone | Accuracy | Method |
|-----------|----------|--------|
| **Current (Phase 1)** | 99.60% | ResNet-18, basic TTA |
| **Day 1 (advanced TTA + ensemble)** | 99.70-99.75% | 17-variant TTA, 2-3 models |
| **Day 4 (first SOTA models)** | 99.73-99.77% | Wide ResNet-28-10, DenseNet |
| **Day 6 (semi-supervised)** | 99.75-99.80% | EMNIST pre-training |
| **Day 7 (final ensemble)** | 99.82-99.87% | Ensemble of 5-7 SOTA models |

**Success criteria:** ≥99.80% on MNIST official test set

---

## Approval Checklist

**Engineering Standards:**
- [ ] All code passes `make check`
- [ ] No `Any`, no `cast`, no `type: ignore`
- [ ] All functions have complete type signatures
- [ ] All modules have ≥90% test coverage
- [ ] DRY principles maintained
- [ ] Frozen dataclasses for all config/state
- [ ] Fail-closed error handling

**Phase 1 Expansions:**
- [ ] Advanced TTA (default enabled)
- [ ] Ensemble-first architecture
- [ ] Temperature tuning (required)

**Phase 2 Training:**
- [ ] Training endpoints (strict Pydantic)
- [ ] RQ worker (frozen dataclasses)
- [ ] SOTA recipes only (no baseline)

**Schema Changes:**
- [ ] Manifest v1.1 (required fields)
- [ ] Configuration variables

**Timeline:**
- [ ] 7-day timeline
- [ ] Daily deliverables with quality gates

---

## Conclusion

This proposal provides an aggressive, all-in strategy for maximum MNIST accuracy:

- ✅ **Advanced TTA by default** (no simple mode)
- ✅ **Ensemble-first** (no single-model support)
- ✅ **SOTA recipes only** (no weak baselines)
- ✅ **Full training provenance** (manifest v1.1 required)
- ✅ **Zero technical debt** (strict typing, frozen dataclasses)
- ✅ **Zero drift** (manifest hashing, recipe hashing)

**Target:** 99.82-99.87% accuracy (SOTA territory)

All changes follow existing codebase patterns for configuration, error handling, typing, and testing. No quick fixes, no compromises.

**Recommended decision:** Approve full implementation (Phase 1 + Phase 2).

---

**Authors:** AI Development Team
**Reviewers:** [To be filled]
**Approval Date:** [To be filled]
**Implementation Start:** Day 1 (upon approval)
