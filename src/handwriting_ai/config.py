from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

_DEFAULT_CONFIG_PATH: Final[Path] = Path("config/handai.toml")


@dataclass(frozen=True)
class AppConfig:
    data_root: Path = Path("/data")
    artifacts_root: Path = Path("/data/artifacts")
    logs_root: Path = Path("/data/logs")
    threads: int = 0
    port: int = 8081


@dataclass(frozen=True)
class DigitsConfig:
    model_dir: Path = Path("/data/digits/models")
    active_model: str = "mnist_resnet18_v1"
    tta: bool = False
    uncertain_threshold: float = 0.70
    max_image_mb: int = 2
    max_image_side_px: int = 1024
    predict_timeout_seconds: int = 5
    visualize_max_kb: int = 16


@dataclass(frozen=True)
class SecurityConfig:
    # Empty string disables check (per design doc)
    api_key: str = ""


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    digits: DigitsConfig
    security: SecurityConfig

    @staticmethod
    def _toml_path() -> Path:
        env_val = os.getenv("HANDWRITING_CONFIG")
        if env_val:
            return Path(env_val)
        return _DEFAULT_CONFIG_PATH

    @classmethod
    def load(cls) -> Settings:
        # Load env first, then override from TOML if present.
        base = cls(
            app=_load_app_from_env(),
            digits=_load_digits_from_env(),
            security=_load_security_from_env(),
        )
        cfg_path = cls._toml_path()
        if not cfg_path.exists():
            return base
        try:
            raw: object = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"Failed to read config TOML: {cfg_path}") from exc
        except Exception as exc:
            raise RuntimeError(f"Invalid TOML config: {cfg_path}") from exc
        app_in: dict[str, object] = _toml_table(raw, "app")
        digits_in_raw: dict[str, object] = _toml_table(raw, "digits")
        security_in: dict[str, object] = _toml_table(raw, "security")
        digits_in = {
            ("uncertain_threshold" if k == "conf_threshold" else k): v
            for k, v in digits_in_raw.items()
        }
        return cls(
            app=_merge_app(base.app, app_in),
            digits=_merge_digits(base.digits, digits_in),
            security=_merge_security(base.security, security_in),
        )


def _load_app_from_env() -> AppConfig:
    a = AppConfig()
    dr = os.getenv("APP__DATA_ROOT")
    ar = os.getenv("APP__ARTIFACTS_ROOT")
    lr = os.getenv("APP__LOGS_ROOT")
    th = os.getenv("APP__THREADS")
    pt = os.getenv("APP__PORT")
    if dr:
        a = replace(a, data_root=Path(dr))
    if ar:
        a = replace(a, artifacts_root=Path(ar))
    if lr:
        a = replace(a, logs_root=Path(lr))
    if th is not None and th.isdigit():
        a = replace(a, threads=int(th))
    if pt is not None and pt.isdigit():
        p = int(pt)
        if not (1 <= p <= 65535):
            raise RuntimeError("APP__PORT out of range")
        a = replace(a, port=p)
    return a


def _load_digits_from_env() -> DigitsConfig:
    d = DigitsConfig()
    md = os.getenv("DIGITS__MODEL_DIR")
    am = os.getenv("DIGITS__ACTIVE_MODEL")
    tta = os.getenv("DIGITS__TTA")
    ut = os.getenv("DIGITS__UNCERTAIN_THRESHOLD")
    mb = os.getenv("DIGITS__MAX_IMAGE_MB")
    mx = os.getenv("DIGITS__MAX_IMAGE_SIDE_PX")
    to = os.getenv("DIGITS__PREDICT_TIMEOUT_SECONDS")
    vk = os.getenv("DIGITS__VISUALIZE_MAX_KB")
    if md:
        d = replace(d, model_dir=Path(md))
    if am:
        d = replace(d, active_model=am)
    if tta is not None:
        d = replace(d, tta=tta.lower() in {"1", "true", "yes"})
    if ut is not None:
        d = replace(d, uncertain_threshold=float(ut))
    if mb is not None:
        d = replace(d, max_image_mb=int(mb))
    if mx is not None:
        d = replace(d, max_image_side_px=int(mx))
    if to is not None:
        d = replace(d, predict_timeout_seconds=int(to))
    if vk is not None:
        d = replace(d, visualize_max_kb=int(vk))
    return d


def _load_security_from_env() -> SecurityConfig:
    s = SecurityConfig()
    key = os.getenv("SECURITY__API_KEY")
    if key is not None:
        s = replace(s, api_key=key)
    return s


def _merge_app(base: AppConfig, data: dict[str, object]) -> AppConfig:
    out = base
    if "data_root" in data:
        out = replace(out, data_root=Path(str(data["data_root"])))
    if "artifacts_root" in data:
        out = replace(out, artifacts_root=Path(str(data["artifacts_root"])))
    if "logs_root" in data:
        out = replace(out, logs_root=Path(str(data["logs_root"])))
    if "threads" in data:
        out = replace(out, threads=int(str(data["threads"])))
    if "port" in data:
        port = int(str(data["port"]))
        if not (1 <= port <= 65535):
            raise RuntimeError("port out of range")
        out = replace(out, port=port)
    return out


def _merge_digits(base: DigitsConfig, data: dict[str, object]) -> DigitsConfig:
    out = base
    if "model_dir" in data:
        out = replace(out, model_dir=Path(str(data["model_dir"])))
    if "active_model" in data:
        out = replace(out, active_model=str(data["active_model"]))
    if "tta" in data:
        out = replace(out, tta=bool(data["tta"]))
    if "uncertain_threshold" in data:
        out = replace(out, uncertain_threshold=float(str(data["uncertain_threshold"])))
    if "max_image_mb" in data:
        out = replace(out, max_image_mb=int(str(data["max_image_mb"])))
    if "max_image_side_px" in data:
        out = replace(out, max_image_side_px=int(str(data["max_image_side_px"])))
    if "predict_timeout_seconds" in data:
        out = replace(out, predict_timeout_seconds=int(str(data["predict_timeout_seconds"])))
    if "visualize_max_kb" in data:
        out = replace(out, visualize_max_kb=int(str(data["visualize_max_kb"])))
    return out


def _toml_table(raw: object, key: str) -> dict[str, object]:
    if isinstance(raw, dict):
        tab: object = raw.get(key, {})
        if isinstance(tab, dict):
            # best-effort copy as object-typed dict
            return {str(k): v for k, v in tab.items()}
    return {}


def _merge_security(base: SecurityConfig, data: dict[str, object]) -> SecurityConfig:
    out = base
    coerced = _coerce_security(data)
    if "api_key" in coerced:
        out = replace(out, api_key=str(coerced["api_key"]))
    return out


def _coerce_security(inp: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    api_key_val = inp.get("api_key")
    if isinstance(api_key_val, str):
        out["api_key"] = api_key_val
    enabled = inp.get("api_key_enabled")
    if isinstance(enabled, bool) and not enabled:
        out["api_key"] = ""
    return out


@dataclass(frozen=True)
class Limits:
    max_bytes: int
    max_side_px: int

    @staticmethod
    def from_settings(s: Settings) -> Limits:
        return Limits(
            max_bytes=int(s.digits.max_image_mb) * 1024 * 1024,
            max_side_px=int(s.digits.max_image_side_px),
        )
