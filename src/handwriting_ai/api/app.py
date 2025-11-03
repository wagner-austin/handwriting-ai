from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable
from typing import Annotated, Protocol

from fastapi import Depends, FastAPI, File, Header, Request, UploadFile
from fastapi.params import Depends as DependsParamType
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile, UnidentifiedImageError

from ..config import Limits, Settings
from ..errors import AppError, ErrorCode, new_error, status_for
from ..inference.engine import InferenceEngine
from ..logging import init_logging, log_event
from ..middleware import RequestIdMiddleware, api_key_dependency
from ..preprocess import PreprocessOptions, run_preprocess
from ..request_context import request_id_var
from ..version import get_version
from .schemas import PredictResponse

ImageFile.LOAD_TRUNCATED_IMAGES = False


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


def _register_read(
    app: FastAPI,
    dep_api_key: Callable[[str | None], None],
    provide_engine: Callable[[], InferenceEngine],
    provide_settings: Callable[[], Settings],
    provide_limits: Callable[[], Limits],
) -> None:
    async def _read_digit(
        file: Annotated[UploadFile, File(...)],
        invert: bool | None = None,
        center: bool = True,
        visualize: bool = False,
        content_length: int | None = Header(default=None, alias="Content-Length"),
    ) -> dict[str, object]:
        engine = provide_engine()
        settings = provide_settings()
        limits = provide_limits()

        ctype = (file.content_type or "").lower()
        if ctype not in ("image/png", "image/jpeg", "image/jpg"):
            raise AppError(
                ErrorCode.unsupported_media_type,
                status_for(ErrorCode.unsupported_media_type),
                "Only PNG and JPEG are supported",
            )

        if content_length is not None and content_length > limits.max_bytes:
            raise AppError(
                ErrorCode.too_large,
                status_for(ErrorCode.too_large),
                "Request body too large",
            )

        raw = await file.read()
        _raise_if_too_large(raw, limits)

        try:
            img = Image.open(io.BytesIO(raw))
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

        w, h = img.size
        if max(w, h) > limits.max_side_px:
            raise AppError(
                ErrorCode.bad_dimensions,
                status_for(ErrorCode.bad_dimensions),
                "Image dimensions too large",
            )

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
) -> FastAPI:
    s = settings or Settings.load()
    init_logging()
    app = FastAPI(title="handwriting-ai", version=get_version().version)
    app.add_middleware(RequestIdMiddleware)

    engine = _create_engine(s)
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

    # Allow explicit injection of a custom engine provider for tests
    _eng_provider = engine_provider if engine_provider is not None else _provide_engine
    _register_read(app, _api_key_dep, _eng_provider, _provide_settings, _provide_limits)

    return app


# Default ASGI app for uvicorn
app = create_app()


class _APIKeyDep(Protocol):
    def __call__(self, x_api_key: str | None) -> None: ...
