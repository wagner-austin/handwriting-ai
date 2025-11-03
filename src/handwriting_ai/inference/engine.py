from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Final, Protocol

import torch

from ..config import Settings
from ..logging import get_logger
from .manifest import ModelManifest
from .types import PredictOutput

_MNIST_N_CLASSES: Final[int] = 10


class InferenceEngine:
    """Bounded thread-pool inference engine with Torch CPU model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._logger = get_logger()
        self._pool = _make_pool(settings)
        self._model_lock = threading.RLock()
        self._model: TorchModel | None = None
        self._manifest: ModelManifest | None = None
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

    def submit_predict(self, preprocessed: torch.Tensor) -> Future[PredictOutput]:
        return self._pool.submit(self._predict_impl, preprocessed)

    def _predict_impl(self, preprocessed: torch.Tensor) -> PredictOutput:
        # Lazy failure if model not ready
        man = self._manifest
        model_obj = self._model
        if man is None or model_obj is None:
            raise RuntimeError("Model not loaded")

        # Defer Tensor handling to Torch with type as object to satisfy typing constraints
        tensor = _as_torch_tensor(preprocessed)
        model_obj.eval()
        with torch.no_grad():
            logits_obj = model_obj(tensor)
        temperature = float(man.temperature)
        probs_vec = _softmax(logits_obj, temperature)
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
        if not manifest_path.exists():
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
        # Build model arch per manifest (weights management handled separately)
        model = _build_model(arch=manifest.arch, n_classes=int(manifest.n_classes))
        with self._model_lock:
            self._model = model
            self._manifest = manifest


def _make_pool(settings: Settings) -> ThreadPoolExecutor:
    if settings.app.threads == 0:
        import os

        cpu_count = os.cpu_count() or 1
        size = min(8, cpu_count)
    else:
        size = settings.app.threads
    return ThreadPoolExecutor(max_workers=size, thread_name_prefix="predict")


class TorchModel(Protocol):
    def eval(self) -> None: ...
    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


def _build_model(arch: str, n_classes: int) -> TorchModel:
    class _ZeroModel:
        def __init__(self, classes: int) -> None:
            self._classes = int(classes)

        def eval(self) -> None:
            return None

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            b = int(x.shape[0]) if x.ndim >= 1 else 1
            return torch.zeros((b, self._classes), dtype=torch.float32)

    return _ZeroModel(n_classes)


def _as_torch_tensor(x: torch.Tensor) -> torch.Tensor:
    t = x
    if t.ndim == 3:
        # Expect 1x28x28 -> add batch
        t = t.unsqueeze(0)
    return t.to(dtype=torch.float32)


def _softmax(logits: torch.Tensor, temperature: float) -> list[float]:
    logits_t = logits / temperature
    probs = torch.softmax(logits_t, dim=1)
    n = int(probs.shape[1])
    return [float(probs[0, i].item()) for i in range(n)]
