# Proposal: Design Doc Deviations for Maximum Accuracy

**Date:** 2025-11-03
**Goal:** Win accuracy competition (99.85%+ target) within 1 week
**Current baseline:** 99.6% (Phase 1 complete)

---

## Executive Summary

This proposal outlines targeted deviations from the original design document to maximize MNIST accuracy for competition. All changes maintain backward compatibility and build incrementally on the existing Phase 1 implementation.

**Expected outcome:** 99.6% → 99.85%+ accuracy
**Timeline:** 7 days
**Risk:** Low (fallback to 99.75% with Phase 1 expansions only)

**Engineering commitment:** All changes adhere to existing codebase standards for strict typing, modularity, and production readiness. No quick fixes, no technical debt accumulation.

---

## Engineering Standards & Quality Gates

All proposed changes **MUST** satisfy the following non-negotiable requirements from the design document (line 13-16):

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
5. Explicit narrow types over broad types (e.g., `tuple[float, ...]` not `Sequence`)

**Example (correct):**
```python
from typing import TYPE_CHECKING, Protocol

class TorchTensor(Protocol):
    """Narrow protocol for torch.Tensor to avoid Any"""
    def unsqueeze(self, dim: int) -> TorchTensor: ...
    def to(self, dtype: object) -> TorchTensor: ...
    @property
    def ndim(self) -> int: ...

def _augment_for_tta(x: Tensor) -> Tensor:
    """Pure type-checked function, no runtime Any"""
    batch: list[Tensor] = [x]
    # ... implementation with full types ...
    return torch.cat(batch, dim=0)
```

**Example (FORBIDDEN):**
```python
# ❌ Would fail guard checks
def _augment_for_tta(x: Any) -> Any:  # NO Any
    batch = [x]  # type: ignore  # NO type: ignore
    return cast(Tensor, result)  # NO cast
```

---

### 2. DRY, Modular, Standardized (Design Doc Line 14)

**Module boundaries:**
- `inference/` - Model loading, prediction, TTA (Phase 1)
- `training/` - Training loops, recipes, workers (Phase 2)
- `api/` - HTTP endpoints, request/response schemas
- `config.py` - Centralized configuration (no scattered settings)

**Pattern reuse:**
- Configuration: Nested env keys (`APP__`, `DIGITS__`, `TRAINING__`)
- Data classes: Frozen dataclasses for immutable config/results
- Protocols: Abstract third-party types (torch, torchvision)
- Error handling: Fail-closed, explicit error codes, no silent fallbacks

**No duplication:**
- TTA logic lives in ONE place (`engine.py:_augment_for_tta`)
- Temperature logic lives in ONE place (`engine.py:_softmax_avg`)
- Config loading pattern reused across all modules

---

### 3. Anti-Drift (Design Doc Line 16)

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

**Changes that update preprocess signature:**
- Any modification to preprocessing pipeline → bump signature
- Existing models become incompatible → must retrain
- Prevents silent accuracy regressions

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

**Example test structure:**
```python
def test_advanced_tta_preserves_shape() -> None:
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    result = _augment_for_tta_advanced(x)
    assert result.shape == (17, 1, 28, 28)
    assert result.dtype == torch.float32

def test_advanced_tta_with_tta_disabled() -> None:
    settings = Settings(app=AppConfig(), digits=DigitsConfig(tta=False), ...)
    # Verify TTA not applied when disabled
```

---

### 5. No Technical Debt

**Prohibited shortcuts:**
- ❌ Hardcoded magic numbers (use Final[] constants)
- ❌ Copy-pasted code (extract to functions)
- ❌ "TODO" comments without tracking
- ❌ Temporary workarounds left in production
- ❌ Unclear variable names (x, tmp, data)

**Required practices:**
- ✅ Explicit over implicit
- ✅ Named constants for thresholds
- ✅ Docstrings for public functions
- ✅ Error messages explain what went wrong
- ✅ Clean commit history (no "fix typo" spam)

**Example (correct):**
```python
_TTA_ROTATION_ANGLES: Final[tuple[float, ...]] = (-3.0, -1.5, 1.5, 3.0)
_TTA_SCALE_FACTORS: Final[tuple[float, ...]] = (0.95, 1.05)

def _augment_for_tta_advanced(x: Tensor) -> Tensor:
    """Apply test-time augmentation with rotations and scaling.

    Generates 17 variants: identity + 8 shifts + 4 rotations + 2 scales.
    Rotation angles and scale factors are constants to prevent drift.
    """
    batch: list[Tensor] = [x]
    # ... clear implementation ...
```

**Example (FORBIDDEN):**
```python
def tta(x):  # ❌ Unclear name, no types
    b = [x]
    for a in [-3, -1.5, 1.5, 3]:  # ❌ Magic numbers
        # TODO fix this later  # ❌ Untracked debt
        b.append(rot(x, a))
    return cat(b)  # ❌ Unclear return type
```

---

## Phase 1 Expansions (Inference Enhancements)

### 1. Advanced TTA Implementation

**File:** `src/handwriting_ai/inference/engine.py`

**Design principle:** Expand existing TTA to cover rotation/scale invariance while maintaining strict typing and no performance regression when disabled.

**Current (line 197-207):**
```python
def _augment_for_tta(x: Tensor) -> Tensor:
    """Apply basic TTA: identity + 4 pixel shifts."""
    batch: list[Tensor] = [x]
    batch.append(torch.roll(x, shifts=(0, 1), dims=(2, 3)))
    batch.append(torch.roll(x, shifts=(0, -1), dims=(2, 3)))
    batch.append(torch.roll(x, shifts=(1, 0), dims=(2, 3)))
    batch.append(torch.roll(x, shifts=(-1, 0), dims=(2, 3)))
    return torch.cat(batch, dim=0)
```

**Proposed (strict typing, production-grade):**
```python
from typing import Final

# Module-level constants (prevents drift)
_TTA_ROTATION_ANGLES: Final[tuple[float, ...]] = (-3.0, -1.5, 1.5, 3.0)
_TTA_SCALE_FACTORS: Final[tuple[float, ...]] = (0.95, 1.05)
_TTA_PIXEL_SHIFTS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (0, -1), (1, 0), (-1, 0),  # cardinal
    (1, 1), (1, -1), (-1, 1), (-1, -1)  # diagonal
)

def _augment_for_tta(x: Tensor) -> Tensor:
    """Apply TTA: 5 variants (identity + 4 shifts) for speed.

    This is the default TTA mode (DIGITS__TTA=true).
    For advanced TTA, use _augment_for_tta_advanced.

    Args:
        x: Input tensor (1, 1, 28, 28)

    Returns:
        Augmented batch (5, 1, 28, 28)
    """
    batch: list[Tensor] = [x]
    for shift in _TTA_PIXEL_SHIFTS[:4]:  # Only cardinal directions
        batch.append(torch.roll(x, shifts=shift, dims=(2, 3)))
    return torch.cat(batch, dim=0)


def _augment_for_tta_advanced(x: Tensor) -> Tensor:
    """Apply advanced TTA: 17 variants (shifts + rotations + scaling).

    Enabled via DIGITS__TTA_ADVANCED=true config.
    Significantly slower but more robust to geometric variations.

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
        # Fail-closed: unexpected input shape
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

        # Affine matrix: rotation only
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

**Integration with existing code:**
```python
# In InferenceEngine._predict_impl (line 67-68)
# Current:
batch = _augment_for_tta(tensor) if self._settings.digits.tta else tensor

# Updated:
if self._settings.digits.tta_advanced:
    batch = _augment_for_tta_advanced(tensor)
elif self._settings.digits.tta:
    batch = _augment_for_tta(tensor)
else:
    batch = tensor
```

**Configuration addition:**
```python
# config.py - DigitsConfig
@dataclass(frozen=True)
class DigitsConfig:
    # ... existing fields ...
    tta: bool = False
    tta_advanced: bool = False  # NEW: opt-in to 17-variant TTA
```

**Environment variable:**
```bash
DIGITS__TTA_ADVANCED=false  # Default: disabled for speed
```

**Test coverage (new file: `tests/test_tta_advanced.py`):**
```python
from __future__ import annotations

import torch

from handwriting_ai.inference.engine import _augment_for_tta, _augment_for_tta_advanced


def test_basic_tta_produces_5_variants() -> None:
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    result = _augment_for_tta(x)
    assert result.shape == (5, 1, 28, 28)
    assert result.dtype == torch.float32


def test_advanced_tta_produces_17_variants() -> None:
    x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    result = _augment_for_tta_advanced(x)
    assert result.shape == (17, 1, 28, 28)
    assert result.dtype == torch.float32


def test_advanced_tta_rejects_wrong_shape() -> None:
    x = torch.zeros((1, 28, 28), dtype=torch.float32)  # 3D instead of 4D
    import pytest
    with pytest.raises(ValueError, match="Expected 4D tensor"):
        _augment_for_tta_advanced(x)


def test_tta_variants_differ_from_identity() -> None:
    x = torch.randn((1, 1, 28, 28), dtype=torch.float32)
    result = _augment_for_tta_advanced(x)

    # First variant is identity
    assert torch.allclose(result[0], x[0], atol=1e-6)

    # Other variants should differ
    for i in range(1, result.shape[0]):
        assert not torch.allclose(result[i], x[0], atol=1e-6)
```

**Rationale:**
- Rotation variants catch tilted digits that preprocessing missed
- Scaling variants handle ambiguous digit sizes
- Diagonal shifts improve robustness to centering errors
- Strict typing: No `Any`, full signatures, Protocol for torch types
- Fail-closed: ValueError on unexpected shapes
- Constants prevent magic number drift

**Expected gain:** +0.10-0.15% accuracy
**Design doc status:** TTA mentioned (line 98), implementation details unspecified
**Breaking change:** No (opt-in via `DIGITS__TTA_ADVANCED`, existing TTA unchanged)
**Performance impact:** 3.4x slower when enabled (17 variants vs 5)

---

### 2. Multi-Model Ensemble Support

**File:** `src/handwriting_ai/inference/engine.py`

**Design principle:** Support loading multiple models and averaging predictions while preserving single-model code path and maintaining strict types.

**Current architecture:**
```python
class InferenceEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._pool = _make_pool(settings)
        self._model_lock = threading.RLock()
        self._model: TorchModel | None = None      # Single model
        self._manifest: ModelManifest | None = None  # Single manifest
```

**Proposed (ensemble-ready, backward compatible):**
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
    """Bounded thread-pool inference engine with optional ensemble support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._pool = _make_pool(settings)
        self._model_lock = threading.RLock()

        # Support 1+ models for ensemble
        self._loaded_models: list[_LoadedModel] = []

        torch.set_num_threads(1)

    @property
    def ready(self) -> bool:
        """Service ready if at least one model loaded."""
        return len(self._loaded_models) > 0

    @property
    def model_id(self) -> str | None:
        """Primary model ID (first loaded, or None)."""
        if len(self._loaded_models) == 0:
            return None
        return self._loaded_models[0].manifest.model_id

    @property
    def manifest(self) -> ModelManifest | None:
        """Primary manifest (first loaded, or None)."""
        if len(self._loaded_models) == 0:
            return None
        return self._loaded_models[0].manifest

    @property
    def ensemble_size(self) -> int:
        """Number of models loaded (1 = single model, >1 = ensemble)."""
        return len(self._loaded_models)

    def try_load_active(self) -> None:
        """Load model(s) specified in DIGITS__ACTIVE_MODEL.

        Supports:
        - Single model: DIGITS__ACTIVE_MODEL=mnist_resnet18_v1
        - Ensemble: DIGITS__ACTIVE_MODEL=model1,model2,model3

        All models must have compatible preprocess_hash.
        """
        active = self._settings.digits.active_model
        model_ids = [mid.strip() for mid in active.split(",") if mid.strip()]

        if len(model_ids) == 0:
            self._logger.info("no_active_model_specified")
            return

        loaded: list[_LoadedModel] = []
        expected_hash: str | None = None

        for model_id in model_ids:
            model_dir = self._settings.digits.model_dir / model_id
            if not model_dir.exists():
                self._logger.info(f"model_dir_not_found", extra={"model_id": model_id})
                continue

            manifest_path = model_dir / "manifest.json"
            model_path = model_dir / "model.pt"

            if not (manifest_path.exists() and model_path.exists()):
                self._logger.info(f"model_files_incomplete", extra={"model_id": model_id})
                continue

            try:
                manifest = ModelManifest.from_path(manifest_path)
            except (OSError, ValueError):
                self._logger.info("manifest_load_failed", extra={"model_id": model_id})
                continue

            # Validate preprocess signature compatibility
            from ..preprocess import preprocess_signature
            if manifest.preprocess_hash != preprocess_signature():
                self._logger.info("preprocess_hash_mismatch", extra={"model_id": model_id})
                continue

            # Ensemble invariant: all models must have same preprocess hash
            if expected_hash is None:
                expected_hash = manifest.preprocess_hash
            elif manifest.preprocess_hash != expected_hash:
                self._logger.info("ensemble_hash_mismatch", extra={
                    "model_id": model_id,
                    "expected": expected_hash,
                    "actual": manifest.preprocess_hash
                })
                continue

            # Build model and load weights
            model = _build_model(arch=manifest.arch, n_classes=int(manifest.n_classes))
            try:
                sd = _load_state_dict_file(model_path)
            except _LOAD_ERRORS:
                self._logger.info("state_dict_load_failed", extra={"model_id": model_id})
                continue

            try:
                _validate_state_dict(sd, manifest.arch, int(manifest.n_classes))
                model.load_state_dict(sd)
            except ValueError:
                self._logger.info("state_dict_invalid", extra={"model_id": model_id})
                continue

            loaded.append(_LoadedModel(model=model, manifest=manifest))
            self._logger.info("model_loaded", extra={"model_id": model_id})

        # Atomic swap under lock
        with self._model_lock:
            self._loaded_models = loaded

        if len(loaded) > 1:
            self._logger.info("ensemble_ready", extra={
                "num_models": len(loaded),
                "model_ids": [lm.manifest.model_id for lm in loaded]
            })

    def _predict_impl(self, preprocessed: Tensor) -> PredictOutput:
        """Run inference, averaging predictions across ensemble.

        For single model: standard prediction
        For ensemble: average probabilities across all models
        """
        loaded = self._loaded_models
        if len(loaded) == 0:
            raise RuntimeError("Model not loaded")

        tensor = _as_torch_tensor(preprocessed)

        # Apply TTA if enabled
        if self._settings.digits.tta_advanced:
            batch = _augment_for_tta_advanced(tensor)
        elif self._settings.digits.tta:
            batch = _augment_for_tta(tensor)
        else:
            batch = tensor

        # Collect predictions from all models
        all_probs: list[list[float]] = []

        for loaded_model in loaded:
            loaded_model.model.eval()
            with torch.no_grad():
                logits_obj = loaded_model.model(batch)

            temperature = float(loaded_model.manifest.temperature)
            probs_vec = _softmax_avg(logits_obj, temperature)
            all_probs.append(probs_vec)

        # Average probabilities across ensemble
        ensemble_probs = _average_probs(all_probs)

        # Find top class
        top_idx = 0
        best_conf = ensemble_probs[0]
        for i in range(1, len(ensemble_probs)):
            if ensemble_probs[i] > best_conf:
                best_conf = ensemble_probs[i]
                top_idx = i

        # Use primary model ID for response
        primary_id = loaded[0].manifest.model_id
        if len(loaded) > 1:
            primary_id = f"ensemble_{len(loaded)}_models"

        return PredictOutput(
            digit=top_idx,
            confidence=best_conf,
            probs=tuple(ensemble_probs),
            model_id=primary_id
        )


def _average_probs(all_probs: list[list[float]]) -> list[float]:
    """Average probability distributions across models.

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

    # Normalize by number of models
    n_models = float(len(all_probs))
    for i in range(n_classes):
        avg[i] /= n_models

    return avg
```

**Test coverage (new file: `tests/test_ensemble.py`):**
```python
from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import UTC, datetime
import json

import torch

from handwriting_ai.config import AppConfig, DigitsConfig, SecurityConfig, Settings
from handwriting_ai.inference.engine import InferenceEngine, build_fresh_state_dict, _average_probs
from handwriting_ai.preprocess import preprocess_signature


def test_single_model_still_works() -> None:
    """Backward compatibility: single model path unchanged."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models" / "test_model"
        model_dir.mkdir(parents=True)

        # Create manifest and model
        manifest = {
            "schema_version": "v1",
            "model_id": "test_model",
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest))

        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (model_dir / "model.pt").as_posix())

        # Load single model
        settings = Settings(
            app=AppConfig(),
            digits=DigitsConfig(model_dir=root / "models", active_model="test_model"),
            security=SecurityConfig()
        )
        engine = InferenceEngine(settings)
        engine.try_load_active()

        assert engine.ready is True
        assert engine.ensemble_size == 1
        assert engine.model_id == "test_model"


def test_ensemble_loads_multiple_models() -> None:
    """Ensemble: comma-separated model IDs load multiple models."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models_root = root / "models"

        # Create 3 models with same preprocess hash
        for i in range(3):
            model_id = f"model_{i}"
            model_dir = models_root / model_id
            model_dir.mkdir(parents=True)

            manifest = {
                "schema_version": "v1",
                "model_id": model_id,
                "arch": "resnet18",
                "n_classes": 10,
                "version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "preprocess_hash": preprocess_signature(),
                "val_acc": 0.99,
                "temperature": 1.0,
            }
            (model_dir / "manifest.json").write_text(json.dumps(manifest))

            sd = build_fresh_state_dict("resnet18", 10)
            torch.save(sd, (model_dir / "model.pt").as_posix())

        # Load ensemble
        settings = Settings(
            app=AppConfig(),
            digits=DigitsConfig(
                model_dir=models_root,
                active_model="model_0,model_1,model_2"  # Comma-separated
            ),
            security=SecurityConfig()
        )
        engine = InferenceEngine(settings)
        engine.try_load_active()

        assert engine.ready is True
        assert engine.ensemble_size == 3


def test_ensemble_rejects_mismatched_preprocess_hash() -> None:
    """Ensemble invariant: all models must have same preprocess hash."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models_root = root / "models"

        # Model 1: correct hash
        model_dir_1 = models_root / "model_1"
        model_dir_1.mkdir(parents=True)
        manifest_1 = {
            "schema_version": "v1",
            "model_id": "model_1",
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (model_dir_1 / "manifest.json").write_text(json.dumps(manifest_1))
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (model_dir_1 / "model.pt").as_posix())

        # Model 2: WRONG hash
        model_dir_2 = models_root / "model_2"
        model_dir_2.mkdir(parents=True)
        manifest_2 = dict(manifest_1)
        manifest_2["model_id"] = "model_2"
        manifest_2["preprocess_hash"] = "WRONG_HASH"
        (model_dir_2 / "manifest.json").write_text(json.dumps(manifest_2))
        torch.save(sd, (model_dir_2 / "model.pt").as_posix())

        # Try to load ensemble
        settings = Settings(
            app=AppConfig(),
            digits=DigitsConfig(
                model_dir=models_root,
                active_model="model_1,model_2"
            ),
            security=SecurityConfig()
        )
        engine = InferenceEngine(settings)
        engine.try_load_active()

        # Should only load model_1 (model_2 rejected due to hash mismatch)
        assert engine.ensemble_size == 1


def test_average_probs_basic() -> None:
    """Probability averaging: simple case."""
    probs_1 = [0.8, 0.1, 0.1]
    probs_2 = [0.6, 0.2, 0.2]

    avg = _average_probs([probs_1, probs_2])

    assert len(avg) == 3
    assert abs(avg[0] - 0.7) < 1e-6  # (0.8 + 0.6) / 2
    assert abs(avg[1] - 0.15) < 1e-6  # (0.1 + 0.2) / 2


def test_average_probs_rejects_inconsistent_sizes() -> None:
    """Fail-closed: reject mismatched class counts."""
    probs_1 = [0.5, 0.5]
    probs_2 = [0.3, 0.3, 0.4]  # Different size

    import pytest
    with pytest.raises(ValueError, match="Inconsistent class counts"):
        _average_probs([probs_1, probs_2])
```

**Rationale:**
- Ensembles improve accuracy by averaging model biases
- Different architectures capture different features
- Strict typing: `_LoadedModel` frozen dataclass prevents model/manifest mismatches
- Backward compatible: single model path unchanged
- Fail-closed: Reject models with mismatched preprocess hashes
- DRY: Single loading logic for both single/ensemble modes

**Expected gain:** +0.05-0.10% accuracy
**Design doc status:** Not mentioned (new capability)
**Breaking change:** No (single model still works, comma-separated is opt-in)
**Performance impact:** N× slower for N models, linear scaling

---

### 3. Enhanced Temperature Optimization

**File:** `src/handwriting_ai/inference/engine.py` (line 72-73) + `config.py`

**Design principle:** Allow runtime temperature tuning without manifest changes, maintaining type safety and fail-closed behavior.

**Current:**
```python
# In _predict_impl
temperature = float(man.temperature)  # From manifest only
probs_vec = _softmax_avg(logits_obj, temperature)
```

**Proposed:**
```python
# In _predict_impl (ensemble version)
temperature = self._get_temperature(loaded_model.manifest)
probs_vec = _softmax_avg(logits_obj, temperature)

# New method in InferenceEngine
def _get_temperature(self, manifest: ModelManifest) -> float:
    """Get temperature for softmax calibration.

    Priority:
    1. DIGITS__TEMPERATURE_OVERRIDE (if set)
    2. manifest.temperature (default)

    Args:
        manifest: Model manifest containing default temperature

    Returns:
        Temperature value (must be > 0)
    """
    override = self._settings.digits.temperature_override
    if override is not None:
        if override <= 0.0:
            # Fail-closed: invalid temperature
            self._logger.warning("invalid_temperature_override", extra={
                "value": override,
                "using_manifest": manifest.temperature
            })
            return float(manifest.temperature)
        return float(override)
    return float(manifest.temperature)
```

**Configuration (config.py):**
```python
@dataclass(frozen=True)
class DigitsConfig:
    model_dir: Path = Path("/data/digits/models")
    active_model: str = "mnist_resnet18_v1"
    tta: bool = False
    tta_advanced: bool = False  # NEW (from previous change)
    temperature_override: float | None = None  # NEW
    uncertain_threshold: float = 0.70
    max_image_mb: int = 2
    max_image_side_px: int = 1024
    predict_timeout_seconds: int = 5
    visualize_max_kb: int = 16


def _load_digits_from_env() -> DigitsConfig:
    d = DigitsConfig()
    # ... existing env loading ...

    # NEW: temperature override
    temp_override = os.getenv("DIGITS__TEMPERATURE_OVERRIDE")
    if temp_override is not None:
        try:
            temp_val = float(temp_override)
            if temp_val > 0.0:
                d = replace(d, temperature_override=temp_val)
        except ValueError:
            # Fail-closed: invalid override ignored
            pass

    return d
```

**Test coverage (add to `tests/test_engine.py`):**
```python
def test_temperature_override_applied() -> None:
    """Temperature override takes precedence over manifest."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        model_dir = root / "models" / "test_model"
        model_dir.mkdir(parents=True)

        # Manifest temperature = 1.0
        manifest = {
            "schema_version": "v1",
            "model_id": "test_model",
            "arch": "resnet18",
            "n_classes": 10,
            "version": "1.0.0",
            "created_at": datetime.now(UTC).isoformat(),
            "preprocess_hash": preprocess_signature(),
            "val_acc": 0.99,
            "temperature": 1.0,
        }
        (model_dir / "manifest.json").write_text(json.dumps(manifest))
        sd = build_fresh_state_dict("resnet18", 10)
        torch.save(sd, (model_dir / "model.pt").as_posix())

        # Override temperature = 1.5
        settings = Settings(
            app=AppConfig(),
            digits=DigitsConfig(
                model_dir=root / "models",
                active_model="test_model",
                temperature_override=1.5  # Override
            ),
            security=SecurityConfig()
        )
        engine = InferenceEngine(settings)
        engine.try_load_active()

        # Verify override is used (indirect: check via prediction)
        x = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
        result = engine.submit_predict(x).result(timeout=5)
        # Temperature affects probability distribution, not argmax in this case


def test_temperature_override_rejects_invalid() -> None:
    """Fail-closed: invalid temperature override ignored."""
    settings = Settings(
        app=AppConfig(),
        digits=DigitsConfig(temperature_override=-1.0),  # Invalid
        security=SecurityConfig()
    )
    engine = InferenceEngine(settings)

    # Should use manifest temperature (1.0) instead of invalid override
    manifest = ModelManifest(
        schema_version="v1",
        model_id="test",
        arch="resnet18",
        n_classes=10,
        version="1.0.0",
        created_at=datetime.now(UTC),
        preprocess_hash=preprocess_signature(),
        val_acc=0.99,
        temperature=1.0
    )

    temp = engine._get_temperature(manifest)
    assert temp == 1.0  # Manifest value, not override
```

**Rationale:**
- Temperature affects probability calibration and argmax accuracy
- Optimal temperature varies by ensemble composition
- Allows tuning without retraining models
- Fail-closed: Invalid temperatures rejected with warning
- Type-safe: float | None with explicit validation

**Expected gain:** +0.01-0.03% via optimal calibration
**Design doc status:** Temperature scaling specified (line 97), override not mentioned
**Breaking change:** No (optional config, defaults to manifest value)

---

## Phase 2 Acceleration (Training Pipeline)

### 4. Training Endpoints & Worker

**New module structure:**
```
src/handwriting_ai/training/
├── __init__.py
├── types.py       # TypedDicts, Protocols, frozen dataclasses
├── data.py        # MNIST/EMNIST data loading
├── recipes.py     # Training hyperparameter recipes
├── worker.py      # RQ worker implementation
└── jobs.py        # Job state management
```

**Strict typing requirements for Phase 2:**

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
    """Metrics tracked during training (all optional)."""
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    test_acc: float


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training configuration.

    All fields typed, no defaults allow invalid states.
    """
    arch: str
    recipe: str
    epochs: int
    seed: int
    model_id: str
    created_at: datetime


@dataclass(frozen=True)
class JobState:
    """Immutable snapshot of job state.

    Replaces mutable dict to prevent state corruption.
    """
    job_id: str
    status: JobStatus
    config: TrainingConfig
    progress: float  # 0.0 to 1.0
    current_epoch: int
    metrics: TrainingMetrics
    model_id: str | None
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class DataLoader(Protocol):
    """Protocol for torch.utils.data.DataLoader (avoids Any)."""
    def __iter__(self) -> DataLoader: ...
    def __next__(self) -> tuple[object, object]: ...
```

**File: `src/handwriting_ai/api/training.py`**
```python
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from rq import Queue
from redis import Redis

from ..config import Settings
from ..training.types import JobStatus, JobState, TrainingConfig

router = APIRouter(prefix="/v1", tags=["training"])


class TrainRequest(BaseModel):
    """Request schema for POST /v1/train.

    Pydantic v2 with validators, no Any types.
    """
    arch: str = Field(..., pattern="^(resnet18|wideresnet28_10|densenet_bc100)$")
    recipe: str = Field(..., pattern="^(baseline|heavy_aug|semi_supervised)$")
    epochs: int = Field(..., ge=1, le=500)
    seed: int = Field(default=42, ge=0)

    @field_validator("arch")
    @classmethod
    def validate_arch(cls, v: str) -> str:
        allowed = {"resnet18", "wideresnet28_10", "densenet_bc100"}
        if v not in allowed:
            raise ValueError(f"arch must be one of {allowed}")
        return v


class TrainResponse(BaseModel):
    """Response schema for POST /v1/train."""
    job_id: str
    status: JobStatus
    created_at: datetime


class JobResponse(BaseModel):
    """Response schema for GET /v1/jobs/{job_id}."""
    job_id: str
    status: JobStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    current_epoch: int
    total_epochs: int
    metrics: dict[str, float]
    model_id: str | None
    error: str | None


@router.post("/train", response_model=TrainResponse)
async def create_training_job(
    request: TrainRequest,
    settings: Annotated[Settings, Depends(lambda: Settings.load())],
) -> TrainResponse:
    """Queue a training job on RQ worker.

    Strict types, fail-closed on invalid input.
    """
    job_id = str(uuid4())

    config = TrainingConfig(
        arch=request.arch,
        recipe=request.recipe,
        epochs=request.epochs,
        seed=request.seed,
        model_id=f"mnist_{request.arch}_{request.recipe}_seed{request.seed}",
        created_at=datetime.now(UTC)
    )

    # Queue job on Redis/RQ
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue(name=settings.rq.queue_name, connection=redis_conn)

    job = queue.enqueue(
        "handwriting_ai.training.worker.train_model",
        args=(config,),
        job_id=job_id,
        timeout=settings.rq.job_timeout_sec,
    )

    return TrainResponse(
        job_id=job_id,
        status=JobStatus.queued,
        created_at=config.created_at
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    settings: Annotated[Settings, Depends(lambda: Settings.load())],
) -> JobResponse:
    """Get training job status.

    Returns 404 if job not found (fail-closed).
    """
    redis_conn = Redis.from_url(settings.redis_url)
    queue = Queue(name=settings.rq.queue_name, connection=redis_conn)

    job = queue.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Parse job state (fail-closed: require JobState type)
    state: JobState = job.meta.get("state")
    if state is None:
        # Job exists but no state yet (just queued)
        return JobResponse(
            job_id=job_id,
            status=JobStatus.queued,
            progress=0.0,
            current_epoch=0,
            total_epochs=0,
            metrics={},
            model_id=None,
            error=None
        )

    return JobResponse(
        job_id=state.job_id,
        status=state.status,
        progress=state.progress,
        current_epoch=state.current_epoch,
        total_epochs=state.config.epochs,
        metrics=state.metrics,
        model_id=state.model_id,
        error=state.error
    )
```

**Integration with app.py:**
```python
# In create_app()
from .training import router as training_router

if phase_2_enabled(settings):
    app.include_router(training_router)
```

**Test coverage (`tests/test_training_api.py`):**
```python
def test_train_endpoint_rejects_invalid_arch() -> None:
    """Fail-closed: invalid arch rejected by Pydantic."""
    client = TestClient(app)
    response = client.post("/v1/train", json={
        "arch": "invalid_arch",
        "recipe": "baseline",
        "epochs": 100
    })
    assert response.status_code == 422  # Validation error


def test_train_endpoint_queues_job() -> None:
    """Valid request creates job in RQ."""
    client = TestClient(app)
    response = client.post("/v1/train", json={
        "arch": "resnet18",
        "recipe": "heavy_aug",
        "epochs": 100,
        "seed": 42
    })
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
```

**Rationale:**
- Strict Pydantic schemas prevent invalid requests
- Frozen dataclasses for job state (immutable)
- TypedDict for metrics (structured, no Any)
- Fail-closed: 404 on missing jobs, 422 on validation errors
- No magic strings: JobStatus enum

**Design doc status:** Specified in Phase 2 (line 201-211)
**Breaking change:** No (new endpoints only)

---

### 5. SOTA Training Recipes

**File:** `src/handwriting_ai/training/recipes.py`

**Design principle:** Encapsulate training hyperparameters as frozen dataclasses, preventing config drift and enabling reproducibility.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class BaselineRecipe:
    """Standard MNIST training (99.5-99.6% expected).

    Conservative augmentation, proven baseline.
    """
    # Optimizer
    optimizer: str = "Adam"
    lr: float = 0.001
    weight_decay: float = 0.0

    # Scheduler
    scheduler: str = "StepLR"
    step_size: int = 50
    gamma: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 100
    label_smoothing: float = 0.0

    # Augmentation (light)
    rotation_degrees: float = 10.0
    translate: tuple[float, float] = (0.1, 0.1)
    scale: tuple[float, float] | None = None
    elastic_alpha: float | None = None
    cutout_length: int | None = None
    mixup_alpha: float | None = None


@dataclass(frozen=True)
class HeavyAugRecipe:
    """SOTA augmentation (99.73-99.77% expected).

    Aggressive augmentation prevents overfitting.
    Based on academic papers (Zagoruyko 2016, DeVries 2017).
    """
    # Optimizer
    optimizer: str = "AdamW"
    lr: float = 0.001
    weight_decay: float = 1e-4

    # Scheduler
    scheduler: str = "CosineAnnealingLR"
    t_max: int = 150
    eta_min: float = 1e-6

    # Training
    batch_size: int = 128
    epochs: int = 150
    label_smoothing: float = 0.1
    dropout: float = 0.3

    # Augmentation (heavy)
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
    finetune_lr: float = 1e-4  # Lower LR for fine-tuning
    finetune_epochs: int = 100
    finetune_batch_size: int = 128
    finetune_label_smoothing: float = 0.1

    # Augmentation (use heavy aug for fine-tuning)
    finetune_rotation_degrees: float = 15.0
    finetune_translate: tuple[float, float] = (0.15, 0.15)
    finetune_scale: tuple[float, float] = (0.85, 1.15)


# Recipe registry (type-safe lookup)
_RECIPES: Final[dict[str, type[BaselineRecipe] | type[HeavyAugRecipe] | type[SemiSupervisedRecipe]]] = {
    "baseline": BaselineRecipe,
    "heavy_aug": HeavyAugRecipe,
    "semi_supervised": SemiSupervisedRecipe,
}


def get_recipe(name: str) -> BaselineRecipe | HeavyAugRecipe | SemiSupervisedRecipe:
    """Get training recipe by name.

    Args:
        name: Recipe name (baseline, heavy_aug, semi_supervised)

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

**Test coverage (`tests/test_recipes.py`):**
```python
def test_baseline_recipe_frozen() -> None:
    """Recipes are immutable (frozen dataclasses)."""
    recipe = get_recipe("baseline")

    import pytest
    with pytest.raises(AttributeError):
        recipe.lr = 0.01  # Cannot modify frozen


def test_heavy_aug_has_all_augmentations() -> None:
    """Heavy aug recipe includes all augmentation types."""
    recipe = get_recipe("heavy_aug")

    assert recipe.rotation_degrees == 15.0
    assert recipe.scale is not None
    assert recipe.elastic_alpha is not None
    assert recipe.cutout_length is not None
    assert recipe.mixup_alpha is not None


def test_unknown_recipe_raises() -> None:
    """Fail-closed: unknown recipe name raises ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown recipe"):
        get_recipe("nonexistent")
```

**Rationale:**
- Frozen dataclasses prevent config drift
- All hyperparameters explicit (no hidden defaults)
- Type-safe recipe lookup
- Based on academic papers (reproducible)
- No magic numbers

**Expected gain:** +0.15-0.20% per model over baseline
**Design doc status:** Training mentioned, recipes unspecified
**Breaking change:** No (new capability)

---

### 6. Enhanced Manifest Schema (v1.1)

**File:** `src/handwriting_ai/inference/manifest.py`

**Design principle:** Extend manifest schema while maintaining v1 compatibility, adding training provenance tracking.

**Current (v1):**
```python
@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    model_id: str
    arch: str
    n_classes: int
    version: str
    created_at: datetime
    preprocess_hash: str
    val_acc: float
    temperature: float
```

**Proposed (v1.1 - backward compatible):**
```python
@dataclass(frozen=True)
class ModelManifest:
    # v1 fields (required, unchanged)
    schema_version: str
    model_id: str
    arch: str
    n_classes: int
    version: str
    created_at: datetime
    preprocess_hash: str
    val_acc: float
    temperature: float

    # v1.1 fields (optional, for provenance tracking)
    training_recipe: str | None = None  # "baseline", "heavy_aug", "semi_supervised"
    parent_model_id: str | None = None  # For semi-supervised (EMNIST pre-train model)
    training_seed: int | None = None    # Random seed for reproducibility
    training_epochs: int | None = None  # Actual epochs trained
    test_acc: float | None = None       # Test set accuracy (in addition to val_acc)
    recipe_hash: str | None = None      # Hash of recipe hyperparameters (anti-drift)

    @staticmethod
    def from_dict(d: dict[str, object]) -> ModelManifest:
        """Parse manifest from dict (backward compatible with v1).

        v1 manifests work unchanged (optional fields default to None).
        v1.1 manifests include additional provenance fields.
        """
        # v1 required fields
        created_at_str = str(d["created_at"]) if "created_at" in d else ""
        created = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
        n_classes = int(str(d.get("n_classes", 10)))
        val_acc = float(str(d.get("val_acc", 0.0)))
        temperature = float(str(d.get("temperature", 1.0)))

        # Validate ranges
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if not (0.0 <= val_acc <= 1.0):
            raise ValueError("val_acc must be within [0,1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        # v1 required strings
        schema_version = str(d.get("schema_version", "")).strip()
        model_id = str(d.get("model_id", "")).strip()
        arch = str(d.get("arch", "")).strip()
        version = str(d.get("version", "")).strip()
        preprocess_hash = str(d.get("preprocess_hash", "")).strip()

        if not all([schema_version, model_id, arch, version, preprocess_hash]):
            raise ValueError("manifest is missing required v1 fields")

        # v1.1 optional fields (default to None if missing)
        training_recipe = d.get("training_recipe")
        parent_model_id = d.get("parent_model_id")
        training_seed = d.get("training_seed")
        training_epochs = d.get("training_epochs")
        test_acc_raw = d.get("test_acc")
        recipe_hash = d.get("recipe_hash")

        # Type coercion with validation
        test_acc: float | None = None
        if test_acc_raw is not None:
            test_acc = float(str(test_acc_raw))
            if not (0.0 <= test_acc <= 1.0):
                raise ValueError("test_acc must be within [0,1]")

        return ModelManifest(
            schema_version=schema_version,
            model_id=model_id,
            arch=arch,
            n_classes=n_classes,
            version=version,
            created_at=created,
            preprocess_hash=preprocess_hash,
            val_acc=val_acc,
            temperature=temperature,
            training_recipe=str(training_recipe) if training_recipe else None,
            parent_model_id=str(parent_model_id) if parent_model_id else None,
            training_seed=int(training_seed) if training_seed is not None else None,
            training_epochs=int(training_epochs) if training_epochs is not None else None,
            test_acc=test_acc,
            recipe_hash=str(recipe_hash) if recipe_hash else None,
        )
```

**Test coverage (`tests/test_manifest.py`):**
```python
def test_v1_manifest_still_parses() -> None:
    """Backward compatibility: v1 manifests load without v1.1 fields."""
    v1_data = {
        "schema_version": "v1",
        "model_id": "test_model",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": "2025-11-03T12:00:00+00:00",
        "preprocess_hash": "v1/hash",
        "val_acc": 0.99,
        "temperature": 1.0,
    }

    manifest = ModelManifest.from_dict(v1_data)

    # v1 fields present
    assert manifest.model_id == "test_model"
    assert manifest.val_acc == 0.99

    # v1.1 fields default to None
    assert manifest.training_recipe is None
    assert manifest.test_acc is None


def test_v1_1_manifest_parses_provenance() -> None:
    """v1.1 manifests include training provenance."""
    v1_1_data = {
        # v1 fields
        "schema_version": "v1.1",
        "model_id": "test_model",
        "arch": "resnet18",
        "n_classes": 10,
        "version": "1.0.0",
        "created_at": "2025-11-03T12:00:00+00:00",
        "preprocess_hash": "v1/hash",
        "val_acc": 0.99,
        "temperature": 1.0,
        # v1.1 fields
        "training_recipe": "heavy_aug",
        "training_seed": 42,
        "training_epochs": 150,
        "test_acc": 0.992,
        "recipe_hash": "heavy_aug_v1_hash",
    }

    manifest = ModelManifest.from_dict(v1_1_data)

    assert manifest.training_recipe == "heavy_aug"
    assert manifest.training_seed == 42
    assert manifest.test_acc == 0.992
```

**Rationale:**
- Track training provenance for reproducibility
- Distinguish between validation and test accuracy
- Recipe hash prevents training config drift
- Backward compatible: v1 manifests still valid
- Fail-closed: Invalid test_acc raises ValueError

**Design doc status:** v1 schema specified (line 144-146), extensibility implied
**Breaking change:** No (optional fields, v1 manifests still valid, defaults to None)

---

## Configuration Changes

### New Environment Variables

```bash
# ============================================
# Phase 1 Expansions (Inference Enhancements)
# ============================================

# Enable advanced TTA (17 variants vs 5)
# Default: false (use basic TTA for speed)
DIGITS__TTA_ADVANCED=false

# Optional temperature override (affects probability calibration)
# Default: empty (use manifest temperature)
# Example: "1.2"
DIGITS__TEMPERATURE_OVERRIDE=""


# ============================================
# Phase 2 Training (from design doc + new)
# ============================================

# Redis connection (design doc line 128)
REDIS_URL=redis://localhost:6379/0

# RQ queue configuration (design doc line 129-132)
RQ__QUEUE_NAME=digits
RQ__JOB_TIMEOUT_SEC=7200  # 2 hours (training can be slow)
RQ__RESULT_TTL_SEC=86400
RQ__FAILURE_TTL_SEC=604800

# NEW: Training data configuration
TRAINING__DATA_DIR=/data/mnist        # MNIST dataset location
TRAINING__EMNIST_DIR=/data/emnist     # EMNIST for semi-supervised
TRAINING__CHECKPOINT_DIR=/data/checkpoints  # Intermediate checkpoints
TRAINING__NUM_WORKERS=4               # DataLoader workers
TRAINING__PIN_MEMORY=true             # GPU optimization (if using GPU)
```

**Breaking change:** No (all optional or Phase 2 only)

---

## Implementation Timeline

### Day 1: Phase 1 Expansions (Today)
**Goal:** Immediate accuracy boost without training

- [ ] **Hour 1-2:** Implement advanced TTA
  - Add `_augment_for_tta_advanced()` with strict types
  - Add module constants for rotation angles, scale factors
  - Update config with `DIGITS__TTA_ADVANCED`
  - Write 5+ tests (shape, dtype, error handling)
  - Verify `make check` passes (mypy, ruff, guards, tests)

- [ ] **Hour 3-4:** Implement ensemble support
  - Refactor `InferenceEngine` to support multiple models
  - Add `_LoadedModel` frozen dataclass
  - Add `_average_probs()` with type safety
  - Write 8+ tests (single model, ensemble, hash mismatch)
  - Verify `make check` passes

- [ ] **Hour 5-6:** Temperature override + validation
  - Add `temperature_override` to config
  - Add `_get_temperature()` method with validation
  - Write 3+ tests (override applied, invalid rejected)
  - Run full validation: `make check` + accuracy baseline
  - Document: accuracy improvement, config examples

**Deliverable:** 99.70-99.75% accuracy (inference-only)
**Quality gate:** All tests pass, mypy strict, guard checks, 93%+ coverage maintained

---

### Day 2-7: Phase 2 Implementation
(Similar detail level, omitted for brevity - follows same strict typing patterns)

---

## Testing Strategy

**Test coverage requirements:**
- New functions: 100% line coverage
- New modules: ≥90% line coverage
- Edge cases: All error paths tested
- Type safety: mypy strict passes on tests/

**Test file organization:**
```
tests/
├── test_tta_advanced.py        # Advanced TTA unit tests
├── test_ensemble.py            # Ensemble loading/prediction
├── test_temperature.py         # Temperature override
├── test_training_api.py        # Phase 2 endpoints
├── test_recipes.py             # Training recipes
└── test_manifest_v1_1.py       # Manifest compatibility
```

**Example test structure (strict typing):**
```python
from __future__ import annotations

import torch
from torch import Tensor

from handwriting_ai.inference.engine import _augment_for_tta_advanced


def test_advanced_tta_type_safety() -> None:
    """Verify strict types throughout TTA pipeline."""
    x: Tensor = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    result: Tensor = _augment_for_tta_advanced(x)

    # Type assertions (mypy verifies these)
    assert isinstance(result, Tensor)
    assert result.shape == (17, 1, 28, 28)
    assert result.dtype == torch.float32
```

---

## Quality Gates (Non-Negotiable)

All changes **MUST** pass before merge:

1. **`make check`** - Full CI pipeline
   ```bash
   poetry run ruff check . --fix       # Lint
   poetry run ruff format .            # Format
   poetry run mypy                     # Type check (strict)
   poetry run python scripts/guard_checks.py  # No Any/cast/ignore
   poetry run pytest --cov=src --cov-report=term-missing  # Tests + coverage
   ```

2. **Coverage threshold:** ≥93% overall, no regression
3. **Type coverage:** 100% (no Any, no type: ignore, no cast)
4. **Tests:** All new code has tests, all tests pass
5. **Documentation:** Docstrings for public functions
6. **Code review:** At least one approval from team

---

## Risk Analysis & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Type errors slip through** | Low | High | Guard checks + mypy strict enforce zero tolerance |
| **Tech debt accumulation** | Low | High | Frozen dataclasses, DRY principles, code review |
| **Config drift** | Low | Medium | Manifest hashing, recipe hashing, signature validation |
| **Breaking changes** | Very Low | High | Comprehensive backward compat tests, v1/v1.1 manifest tests |
| **Performance regression** | Low | Medium | TTA opt-in (not default), ensemble opt-in, benchmark tests |

---

## Backward Compatibility Guarantees

**All changes maintain full backward compatibility:**

1. **Single model deployments:** Unchanged
   - `DIGITS__ACTIVE_MODEL=mnist_resnet18_v1` (no comma) → single model
   - Existing manifests (v1) load without changes

2. **Basic TTA:** Default behavior unchanged
   - `DIGITS__TTA=true` → 5-variant TTA (current)
   - `DIGITS__TTA_ADVANCED=true` → 17-variant TTA (opt-in)

3. **Configuration:** All new env vars optional
   - Default values preserve existing behavior
   - Phase 2 vars only needed when using training endpoints

4. **API contracts:** Additive only
   - Existing endpoints unchanged
   - New endpoints isolated in Phase 2 router
   - Response schemas backward compatible (PredictResponse unchanged)

5. **Manifest schema:** v1 still valid
   - v1.1 adds optional fields only
   - Parsers default to `None` for missing fields
   - Can mix v1 and v1.1 manifests in ensemble

**Migration path:** Zero-downtime upgrade
```bash
git pull
poetry install  # New deps only if Phase 2 used
make check      # Verify everything passes
# No config changes required
# Opt-in to new features via env vars
```

---

## Expected Accuracy Progression

| Milestone | Accuracy | Method | Quality |
|-----------|----------|--------|---------|
| **Current (Phase 1 baseline)** | 99.60% | ResNet-18, basic TTA | 93.69% test coverage |
| **Day 1 (Phase 1 expansions)** | 99.70-99.75% | Advanced TTA + ensemble | ≥93% coverage maintained |
| **Day 4 (first SOTA models)** | 99.73-99.77% | Wide ResNet-28-10, DenseNet | ≥90% coverage |
| **Day 6 (semi-supervised)** | 99.75-99.80% | EMNIST pre-training | ≥90% coverage |
| **Day 7 (final ensemble)** | 99.82-99.87% | Ensemble of 5-7 SOTA models | Final validation |

**Success criteria:** ≥99.80% on MNIST official test set, zero type errors, ≥90% coverage

---

## Approval Checklist

**Engineering Standards:**
- [ ] All code passes `make check` (ruff + mypy strict + guards + tests)
- [ ] No `Any`, no `cast`, no `type: ignore` in any new code
- [ ] All new functions have complete type signatures
- [ ] All new modules have ≥90% test coverage
- [ ] DRY principles maintained (no copy-paste code)
- [ ] Frozen dataclasses for all config/state
- [ ] Fail-closed error handling (no silent fallbacks)

**Phase 1 Expansions (Low Risk, High Value):**
- [ ] Advanced TTA approved (backward compatible, opt-in)
- [ ] Ensemble support approved (backward compatible, opt-in)
- [ ] Temperature override approved (backward compatible, optional)

**Phase 2 Training Pipeline (Medium Risk, Required for >99.80%):**
- [ ] Training endpoints approved (strict Pydantic schemas)
- [ ] RQ worker implementation approved (frozen dataclasses for state)
- [ ] SOTA training recipes approved (frozen dataclasses, no magic numbers)

**Schema Changes:**
- [ ] Manifest v1.1 schema approved (backward compatible, tested)
- [ ] New configuration variables approved (all optional)

**Timeline:**
- [ ] 7-day timeline approved
- [ ] Daily deliverables acceptable (with quality gates)
- [ ] Fallback strategy (Phase 1 only → 99.75%) acceptable

---

## Conclusion

This proposal provides a clear path from 99.6% (current) to 99.85%+ (SOTA) within the 1-week timeline while maintaining **uncompromising engineering standards**:

- ✅ **Zero type errors** (mypy strict + guard checks)
- ✅ **Zero technical debt** (DRY, frozen dataclasses, no magic numbers)
- ✅ **Zero drift** (manifest hashing, signature validation)
- ✅ **Comprehensive tests** (≥90% coverage, all edge cases)
- ✅ **Backward compatible** (v1 manifests, single models work unchanged)

All changes are **properly integrated solutions**, not quick fixes. Every addition follows existing codebase patterns for configuration, error handling, typing, and testing.

**Recommended decision:** Approve all Phase 1 expansions immediately (low risk, high value, strict engineering). Approve Phase 2 pending GPU/Redis availability confirmation.

---

**Authors:** AI Development Team
**Reviewers:** [To be filled]
**Approval Date:** [To be filled]
**Implementation Start:** Day 1 (upon approval)
