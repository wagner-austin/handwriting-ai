from __future__ import annotations

import logging
import pickle
import threading
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import torch
from torch import Tensor

from ..config import Settings
from ..logging import get_logger
from .manifest import ModelManifest
from .types import PredictOutput

_MNIST_N_CLASSES: Final[int] = 10
_LOAD_ERRORS: Final[tuple[type[BaseException], ...]] = (
    OSError,
    ValueError,
    RuntimeError,
    TypeError,
    EOFError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


class InferenceEngine:
    """Bounded thread-pool inference engine with Torch CPU model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._pool = _make_pool(settings)
        self._model_lock = threading.RLock()
        self._model: TorchModel | None = None
        self._manifest: ModelManifest | None = None
        self._artifacts_dir: Path | None = None
        self._last_manifest_mtime: float | None = None
        self._last_model_mtime: float | None = None
        torch.set_num_threads(1)

    @property
    def ready(self) -> bool:
        return self._model is not None and self._manifest is not None

    @property
    def model_id(self) -> str | None:
        return self._manifest.model_id if self._manifest is not None else None

    @property
    def manifest(self) -> ModelManifest | None:
        return self._manifest

    def submit_predict(self, preprocessed: Tensor) -> Future[PredictOutput]:
        return self._pool.submit(self._predict_impl, preprocessed)

    def _predict_impl(self, preprocessed: Tensor) -> PredictOutput:
        # Lazy failure if model not ready
        man = self._manifest
        model_obj = self._model
        if man is None or model_obj is None:
            raise RuntimeError("Model not loaded")

        # Defer Tensor handling to Torch with type as object to satisfy typing constraints
        tensor = _as_torch_tensor(preprocessed)
        # Test-time augmentation: average predictions across small shifts
        batch = _augment_for_tta(tensor) if self._settings.digits.tta else tensor
        model_obj.eval()
        with torch.no_grad():
            logits_obj = model_obj(batch)
        temperature = float(man.temperature)
        probs_vec = _softmax_avg(logits_obj, temperature)
        probs_py = tuple(float(x) for x in probs_vec)
        top_idx = 0
        best = probs_py[0]
        for i in range(1, len(probs_py)):
            if probs_py[i] > best:
                best = probs_py[i]
                top_idx = i
        conf = float(probs_py[top_idx])
        return PredictOutput(digit=top_idx, confidence=conf, probs=probs_py, model_id=man.model_id)

    def try_load_active(self) -> None:
        active = self._settings.digits.active_model
        model_dir = self._settings.digits.model_dir / active
        if not model_dir.exists():
            return
        manifest_path = model_dir / "manifest.json"
        model_path = model_dir / "model.pt"
        if not (manifest_path.exists() and model_path.exists()):
            return
        try:
            manifest = ModelManifest.from_path(manifest_path)
        except (OSError, ValueError):
            self._logger.info("manifest_load_failed")
            return
        # Validate preprocess signature compatibility
        from ..preprocess import preprocess_signature

        if manifest.preprocess_hash != preprocess_signature():
            # Refuse to load incompatible model
            return
        # Build model arch per manifest and load weights strictly
        model = _build_model(arch=manifest.arch, n_classes=int(manifest.n_classes))
        try:
            sd = _load_state_dict_file(model_path)
        except _LOAD_ERRORS:
            logging.getLogger("handwriting_ai").info("state_dict_load_failed")
            return
        try:
            _validate_state_dict(sd, manifest.arch, int(manifest.n_classes))
            model.load_state_dict(sd)
        except ValueError:
            self._logger.info("state_dict_invalid")
            return
        with self._model_lock:
            self._model = model
            self._manifest = manifest
            self._artifacts_dir = model_dir
            try:
                self._last_manifest_mtime = manifest_path.stat().st_mtime
                self._last_model_mtime = model_path.stat().st_mtime
            except OSError:
                # If mtimes cannot be read, hot-reload will be disabled.
                self._logger.info("artifact_mtime_unavailable")
                self._last_manifest_mtime = None
                self._last_model_mtime = None

    def reload_if_changed(self) -> bool:
        """Reload active model if manifest or weights changed on disk.

        Returns True if a reload occurred and engine remains ready; False otherwise.
        """
        art = self._artifacts_dir
        if art is None:
            return False
        manifest_path = (art / "manifest.json") if art else None
        model_path = (art / "model.pt") if art else None
        if manifest_path is None or model_path is None:
            return False
        try:
            m1 = manifest_path.stat().st_mtime
            m2 = model_path.stat().st_mtime
        except OSError:
            self._logger.info("artifact_mtime_unavailable")
            return False
        if self._last_manifest_mtime is None or self._last_model_mtime is None:
            return False
        if m1 <= self._last_manifest_mtime and m2 <= self._last_model_mtime:
            return False
        # Attempt reload via normal path
        self.try_load_active()
        return self.ready


def _make_pool(settings: Settings) -> ThreadPoolExecutor:
    if settings.app.threads == 0:
        import os

        cpu_count = os.cpu_count() or 1
        size = min(8, cpu_count)
    else:
        size = settings.app.threads
    return ThreadPoolExecutor(max_workers=size, thread_name_prefix="predict")


class TorchModel(Protocol):
    def eval(self) -> object: ...
    def __call__(self, x: Tensor) -> Tensor: ...
    def load_state_dict(self, sd: dict[str, Tensor]) -> object: ...


if TYPE_CHECKING:

    def _build_model(arch: str, n_classes: int) -> TorchModel: ...
else:

    def _build_model(arch: str, n_classes: int) -> TorchModel:
        import importlib

        import torch.nn as nn

        tv_models = importlib.import_module("torchvision.models")
        fn_obj = getattr(tv_models, "resnet18", None)
        if not callable(fn_obj):
            raise RuntimeError("torchvision.models.resnet18 is not callable")
        inner = fn_obj(weights=None, num_classes=int(n_classes))
        # CIFAR-style stem and 1-channel, if attributes exist
        if hasattr(inner, "conv1"):
            inner.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if hasattr(inner, "maxpool"):
            inner.maxpool = nn.Identity()
        return inner


if TYPE_CHECKING:

    def build_fresh_state_dict(arch: str, n_classes: int) -> dict[str, Tensor]: ...
else:

    def build_fresh_state_dict(arch: str, n_classes: int) -> dict[str, Tensor]:
        m = _build_model(arch=arch, n_classes=n_classes)
        sd_obj = m.state_dict()
        if not isinstance(sd_obj, dict):
            raise RuntimeError("state_dict() did not return a dict")
        out: dict[str, Tensor] = {}
        for k, v in sd_obj.items():
            if isinstance(k, str) and torch.is_tensor(v):
                out[k] = v
            else:
                raise RuntimeError("invalid state dict entry from model")
        return out


def _as_torch_tensor(x: Tensor) -> Tensor:
    t = x
    if t.ndim == 3:
        # Expect 1x28x28 -> add batch
        t = t.unsqueeze(0)
    return t.to(dtype=torch.float32)


def _softmax_avg(logits: Tensor, temperature: float) -> list[float]:
    logits_t = logits / temperature
    probs = torch.softmax(logits_t, dim=1)
    mean_probs = probs.mean(dim=0) if probs.ndim == 2 and probs.shape[0] > 1 else probs[0]
    n = int(mean_probs.shape[0])
    return [float(mean_probs[i].item()) for i in range(n)]


def _augment_for_tta(x: Tensor) -> Tensor:
    # Input is 4D (1,1,28,28)
    if x.ndim != 4:
        return x
    # Identity + small shifts
    batch = [x]
    batch.append(torch.roll(x, shifts=(0, 1), dims=(2, 3)))  # right by 1
    batch.append(torch.roll(x, shifts=(0, -1), dims=(2, 3)))  # left by 1
    batch.append(torch.roll(x, shifts=(1, 0), dims=(2, 3)))  # down by 1
    batch.append(torch.roll(x, shifts=(-1, 0), dims=(2, 3)))  # up by 1
    return torch.cat(batch, dim=0)


if TYPE_CHECKING:

    def _load_state_dict_file(path: Path) -> dict[str, Tensor]: ...
else:

    def _load_state_dict_file(path: Path) -> dict[str, Tensor]:
        obj = torch.load(path.as_posix(), map_location=torch.device("cpu"), weights_only=True)
        sd_obj = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        if not isinstance(sd_obj, dict):
            raise ValueError("state dict file did not contain a dict")
        out: dict[str, Tensor] = {}
        for k, v in sd_obj.items():
            if isinstance(k, str) and torch.is_tensor(v):
                out[k] = v
            else:
                raise ValueError("invalid state dict entry")
        return out


def _validate_state_dict(sd: dict[str, Tensor], arch: str, n_classes: int) -> None:
    w = sd.get("fc.weight")
    b = sd.get("fc.bias")
    if w is None or b is None:
        raise ValueError("missing classifier weights in state dict")
    if w.ndim != 2 or b.ndim != 1:
        raise ValueError("invalid classifier tensor dimensions")
    if int(w.shape[0]) != n_classes or int(b.shape[0]) != n_classes:
        raise ValueError("classifier head size does not match n_classes")
    # ResNet-18 expected feature dimension
    expected_in = 512
    if int(w.shape[1]) != expected_in:
        raise ValueError("classifier head in_features does not match backbone")
    # Minimal backbone invariants for resnet18 CIFAR-style stem
    conv1 = sd.get("conv1.weight")
    if conv1 is None or conv1.ndim != 4:
        raise ValueError("missing or invalid conv1.weight")
    if int(conv1.shape[0]) != 64 or int(conv1.shape[1]) != 1:
        raise ValueError("unexpected conv1 shape for 1-channel stem")
    # Expect presence of top-level batch norm
    if "bn1.weight" not in sd or "bn1.bias" not in sd:
        raise ValueError("missing bn1 parameters")
    # Ensure main layers exist
    has_layers = all(any(k.startswith(f"layer{i}.") for k in sd) for i in range(1, 5))
    if not has_layers:
        raise ValueError("missing resnet layer blocks")
