from __future__ import annotations

import base64
import io
import logging
import pickle
import threading
import time
from collections.abc import Callable
from typing import Annotated, Protocol
from weakref import WeakKeyDictionary

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.params import Depends as DependsParamType
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile, UnidentifiedImageError
from starlette.datastructures import FormData

from ..config import Limits, Settings
from ..errors import AppError, ErrorCode, new_error, status_for
from ..inference.engine import InferenceEngine
from ..inference.engine import (
    _load_state_dict_file as _engine_load_state_dict_file,
)
from ..inference.engine import (
    _validate_state_dict as _engine_validate_state_dict,
)
from ..inference.manifest import ModelManifest
from ..logging import init_logging, log_event
from ..middleware import RequestIdMiddleware, api_key_dependency
from ..preprocess import PreprocessOptions, run_preprocess
from ..request_context import request_id_var
from ..version import get_version
from .schemas import PredictResponse

ImageFile.LOAD_TRUNCATED_IMAGES = False

# Defaults for admin upload endpoint parameters (satisfy Ruff B008).
# Use distinct sentinels for separate UploadFile parameters to avoid object aliasing.
_form_model_id: str = Form(...)
_form_activate: bool = Form(False)
_file_required_manifest: UploadFile = File(...)
_file_required_model: UploadFile = File(...)


def _setup_optional_reloader(
    app: FastAPI, engine: InferenceEngine, reload_interval_seconds: float | None
) -> None:
    """Optionally attach a background reloader for model artifacts.

    If `reload_interval_seconds` is falsy or non-positive, no handlers are added.
    """
    if reload_interval_seconds is None or float(reload_interval_seconds) <= 0.0:
        return

    stop_evt: threading.Event | None = None
    thread: threading.Thread | None = None

    def _start_bg_reloader() -> None:
        nonlocal stop_evt, thread
        stop_evt = threading.Event()

        def _loop() -> None:
            interval = float(reload_interval_seconds)
            # Use wait on the event to allow prompt shutdown
            while not stop_evt.is_set():
                engine.reload_if_changed()
                stop_evt.wait(interval)

        thread = threading.Thread(target=_loop, name="model-reloader", daemon=True)
        thread.start()

    def _stop_bg_reloader() -> None:
        nonlocal stop_evt, thread
        if stop_evt is not None:
            stop_evt.set()
        if thread is not None:
            thread.join(timeout=1.0)

    app.add_event_handler("startup", _start_bg_reloader)
    app.add_event_handler("shutdown", _stop_bg_reloader)
    # Expose handlers for white-box tests via a weak map keyed by app instance
    _RELOADER_HANDLES[app] = (_start_bg_reloader, _stop_bg_reloader)


_RELOADER_HANDLES: WeakKeyDictionary[FastAPI, tuple[Callable[[], None], Callable[[], None]]] = (
    WeakKeyDictionary()
)


def _debug_invoke_reloader_start(app: FastAPI) -> None:
    pair = _RELOADER_HANDLES.get(app)
    if pair is not None:
        start, _ = pair
        start()


def _debug_invoke_reloader_stop(app: FastAPI) -> None:
    pair = _RELOADER_HANDLES.get(app)
    if pair is not None:
        _, stop = pair
        stop()


# Exception handlers (module-level to keep app factory simple)
async def _handle_app_error(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):
        # Fallback to generic handler path
        rid = request_id_var.get()
        body = new_error(ErrorCode.internal_error, rid, message=str(exc))
        return JSONResponse(status_code=500, content=body.to_dict())
    rid = request_id_var.get()
    body = new_error(exc.code, rid, message=exc.message)
    return JSONResponse(status_code=exc.http_status, content=body.to_dict())


async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    rid = request_id_var.get()
    body = new_error(ErrorCode.internal_error, rid, message="Internal server error.")
    return JSONResponse(status_code=500, content=body.to_dict())


def _create_engine(settings: Settings) -> InferenceEngine:
    engine = InferenceEngine(settings)
    engine.try_load_active()
    return engine


def _register_basic(app: FastAPI, engine: InferenceEngine) -> None:
    async def _healthz() -> dict[str, str]:
        return {"status": "ok"}

    async def _readyz() -> dict[str, object]:
        man = engine.manifest
        if engine.ready and man is not None:
            return {"status": "ready"}
        return {
            "status": "not_ready",
            "model_loaded": engine.ready,
            "model_id": engine.model_id,
            "manifest_schema_version": (man.schema_version if man is not None else None),
            "build": get_version().build,
        }

    async def _version() -> dict[str, object]:
        v = get_version()
        return {"service": v.service, "version": v.version, "build": v.build, "commit": v.commit}

    app.add_api_route("/healthz", _healthz, methods=["GET"])
    app.add_api_route("/readyz", _readyz, methods=["GET"])
    app.add_api_route("/version", _version, methods=["GET"])


def _register_models(app: FastAPI, engine: InferenceEngine) -> None:
    async def _model_active() -> dict[str, object]:
        man = engine.manifest
        if man is None:
            return {"model_loaded": False, "model_id": None}
        return {
            "model_loaded": True,
            "model_id": man.model_id,
            "arch": man.arch,
            "n_classes": man.n_classes,
            "version": man.version,
            "created_at": man.created_at.isoformat(),
            "schema_version": man.schema_version,
            "val_acc": man.val_acc,
            "temperature": man.temperature,
        }

    app.add_api_route("/v1/models/active", _model_active, methods=["GET"])


def _raise_if_too_large(raw: bytes, limits: Limits) -> None:
    if len(raw) > limits.max_bytes:
        raise AppError(
            ErrorCode.too_large,
            status_for(ErrorCode.too_large),
            "File exceeds size limit",
        )


def _strict_validate_multipart(form: FormData) -> None:
    # Reject any unexpected form fields
    for key in form:
        if key != "file":
            raise AppError(
                ErrorCode.malformed_multipart,
                status_for(ErrorCode.malformed_multipart),
                "Unexpected form field",
            )
    # Require exactly one file part
    n_files = len(form.getlist("file"))
    if n_files != 1:
        raise AppError(
            ErrorCode.malformed_multipart,
            status_for(ErrorCode.malformed_multipart),
            "Multiple file parts not allowed" if n_files > 1 else "Missing file part",
        )


def _open_image_bytes(raw: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as _err:
        raise AppError(
            ErrorCode.invalid_image,
            status_for(ErrorCode.invalid_image),
            "Failed to decode image",
        ) from None
    except Image.DecompressionBombError as _err:
        raise AppError(
            ErrorCode.too_large,
            status_for(ErrorCode.too_large),
            "Decompression bomb triggered",
        ) from None


def _validate_image_dimensions(img: Image.Image, limits: Limits) -> None:
    w, h = img.size
    if max(w, h) > limits.max_side_px:
        raise AppError(
            ErrorCode.bad_dimensions,
            status_for(ErrorCode.bad_dimensions),
            "Image dimensions too large",
        )


def _ensure_supported_content_type(ctype: str) -> None:
    if ctype not in ("image/png", "image/jpeg", "image/jpg"):
        raise AppError(
            ErrorCode.unsupported_media_type,
            status_for(ErrorCode.unsupported_media_type),
            "Only PNG and JPEG are supported",
        )


def _register_read(
    app: FastAPI,
    dep_api_key: Callable[[str | None], None],
    provide_engine: Callable[[], InferenceEngine],
    provide_settings: Callable[[], Settings],
    provide_limits: Callable[[], Limits],
) -> None:
    async def _read_digit(
        request: Request,
        file: Annotated[UploadFile, File(...)],
        invert: bool | None = None,
        center: bool = True,
        visualize: bool = False,
        content_length: int | None = Header(default=None, alias="Content-Length"),
    ) -> dict[str, object]:
        engine = provide_engine()
        settings = provide_settings()
        limits = provide_limits()

        # Enforce strict multipart structure: exactly one 'file' part and no extras.
        form = await request.form()
        _strict_validate_multipart(form)
        ctype = (file.content_type or "").lower()
        _ensure_supported_content_type(ctype)

        if content_length is not None and content_length > limits.max_bytes:
            raise AppError(
                ErrorCode.too_large,
                status_for(ErrorCode.too_large),
                "Request body too large",
            )

        raw = await file.read()
        _raise_if_too_large(raw, limits)
        img = _open_image_bytes(raw)

        _validate_image_dimensions(img, limits)

        opts = PreprocessOptions(
            invert=invert,
            center=center,
            visualize=visualize,
            visualize_max_kb=int(settings.digits.visualize_max_kb),
        )

        t0 = time.perf_counter()
        pre = run_preprocess(img, opts)

        from concurrent.futures import TimeoutError as _FutTimeout

        fut = engine.submit_predict(pre.tensor)
        try:
            out = fut.result(timeout=float(settings.digits.predict_timeout_seconds))
        except _FutTimeout:
            fut.cancel()
            raise AppError(
                ErrorCode.timeout,
                status_for(ErrorCode.timeout),
                "Prediction timed out",
            ) from None
        except RuntimeError as _err:
            # Surface a clear error when engine has no model loaded
            msg = str(_err)
            if "Model not loaded" in msg:
                raise AppError(
                    ErrorCode.service_not_ready,
                    status_for(ErrorCode.service_not_ready),
                    "Model not loaded. Upload or train a model.",
                ) from None
            raise

        dt_ms = int((time.perf_counter() - t0) * 1000.0)

        uncertain = out.confidence < float(settings.digits.uncertain_threshold)
        visual_b64: str | None = (
            base64.b64encode(pre.visual_png).decode("ascii") if pre.visual_png else None
        )
        # Structured log for successful read
        log_event(
            "read_finished",
            fields={
                "latency_ms": dt_ms,
                "digit": int(out.digit),
                "confidence": float(out.confidence),
                "model_id": out.model_id,
                "uncertain": bool(uncertain),
            },
        )

        return {
            "digit": int(out.digit),
            "confidence": float(out.confidence),
            "probs": [float(p) for p in out.probs],
            "model_id": out.model_id,
            "visual_png_b64": visual_b64,
            "uncertain": bool(uncertain),
            "latency_ms": dt_ms,
        }

    api_dep: DependsParamType = Depends(dep_api_key)

    app.add_api_route(
        "/v1/read",
        _read_digit,
        methods=["POST"],
        response_model=PredictResponse,
        dependencies=[api_dep],
    )
    app.add_api_route(
        "/v1/predict",
        _read_digit,
        methods=["POST"],
        response_model=PredictResponse,
        dependencies=[api_dep],
    )


def create_app(
    settings: Settings | None = None,
    engine_provider: Callable[[], InferenceEngine] | None = None,
    *,
    reload_interval_seconds: float | None = None,
) -> FastAPI:
    """Application factory.

    Parameters:
    - `settings`: Optional pre-loaded settings; when omitted, loads defaults.
    - `engine_provider`: Optional provider for a custom `InferenceEngine` (primarily for tests).
    - `reload_interval_seconds`: When provided and > 0, starts a background thread on startup
      that periodically calls `engine.reload_if_changed()` to pick up model artifact changes.
    """
    s = settings or Settings.load()
    init_logging()
    app = FastAPI(title="handwriting-ai", version=get_version().version)
    app.add_middleware(RequestIdMiddleware)

    # Build shared engine instance (or use provided one) used across routes and reloader
    engine: InferenceEngine = (
        engine_provider() if engine_provider is not None else _create_engine(s)
    )
    limits = Limits.from_settings(s)
    _api_key_dep = api_key_dependency(s)

    # Exception handlers
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(Exception, _handle_unexpected)

    def _provide_engine() -> InferenceEngine:
        return engine

    def _provide_settings() -> Settings:
        return s

    def _provide_limits() -> Limits:
        return limits

    # Expose providers for dependency overrides in tests
    app.state.provide_engine = _provide_engine
    app.state.provide_settings = _provide_settings
    app.state.provide_limits = _provide_limits

    _register_basic(app, engine)
    _register_models(app, engine)
    adm_dep: DependsParamType = Depends(_api_key_dep)
    _register_admin(app, engine, adm_dep, _provide_settings)
    # Route registrations and dependencies
    _register_read(app, _api_key_dep, _provide_engine, _provide_settings, _provide_limits)

    # Optional background reloader for model artifacts
    _setup_optional_reloader(app, engine, reload_interval_seconds)

    return app


class _APIKeyDep(Protocol):
    def __call__(self, x_api_key: str | None) -> None: ...


def _register_admin(
    app: FastAPI,
    engine: InferenceEngine,
    dep_api_key: DependsParamType,
    provide_settings: Callable[[], Settings],
) -> None:
    async def _upload_model(
        model_id: str = _form_model_id,
        activate: bool = _form_activate,
        manifest: UploadFile = _file_required_manifest,
        model: UploadFile = _file_required_model,
    ) -> dict[str, object]:
        logging.getLogger("handwriting_ai").info(
            f"upload_files_received manifest.filename={manifest.filename} "
            f"model.filename={model.filename} same_object={manifest is model}"
        )
        man_bytes = await manifest.read()
        try:
            man = ModelManifest.from_json(man_bytes.decode("utf-8"))
        except ValueError:
            raise AppError(ErrorCode.preprocessing_failed, 400, "Invalid manifest") from None
        from ..preprocess import preprocess_signature as _sig

        if man.preprocess_hash != _sig():
            raise AppError(ErrorCode.preprocessing_failed, 400, "Preprocess signature mismatch")
        if man.model_id != model_id:
            raise AppError(ErrorCode.preprocessing_failed, 400, "Model id mismatch")

        s = provide_settings()
        dest = s.digits.model_dir / model_id
        dest.mkdir(parents=True, exist_ok=True)
        model_bytes = await model.read()
        logging.getLogger("handwriting_ai").info(
            f"admin_upload_received model_bytes={len(model_bytes)} manifest_bytes={len(man_bytes)}"
        )
        (dest / "manifest.json").write_text(man_bytes.decode("utf-8"), encoding="utf-8")
        (dest / "model.pt").write_bytes(model_bytes)
        # Verify write completed successfully
        written_size = (dest / "model.pt").stat().st_size
        logging.getLogger("handwriting_ai").info(f"admin_upload_written size_bytes={written_size}")

        # Two-phase validation per design:
        # - When activate=True: strictly validate by loading the state dict and checking shape,
        #   then optionally reload the active engine.
        # - When activate=False: ensure the model file is non-empty (transport sanity) but do not
        #   attempt to parse weights; the model may be validated at activation time.
        if activate:
            load_errors = (RuntimeError, ValueError, TypeError, OSError, pickle.UnpicklingError)
            try:
                sd = _engine_load_state_dict_file(dest / "model.pt")
                _engine_validate_state_dict(sd, man.arch, int(man.n_classes))
            except load_errors:
                raise AppError(
                    ErrorCode.invalid_model,
                    status_for(ErrorCode.invalid_model),
                    "Invalid model file",
                ) from None

            if model_id == s.digits.active_model:
                try:
                    engine.try_load_active()
                except (RuntimeError, ValueError, OSError, TypeError):
                    logging.getLogger("handwriting_ai").info("admin_reload_failed")
        else:
            # Basic transport-level check: avoid persisting empty artifacts
            if written_size <= 0:
                raise AppError(
                    ErrorCode.invalid_model,
                    status_for(ErrorCode.invalid_model),
                    "Invalid model file",
                )

        out: dict[str, object] = {
            "ok": True,
            "model_id": model_id,
            "run_id": man.created_at.isoformat(),
        }
        return out

    app.add_api_route(
        "/v1/admin/models/upload",
        _upload_model,
        methods=["POST"],
        dependencies=[dep_api_key],
    )


# Default ASGI app for uvicorn
app = create_app()
