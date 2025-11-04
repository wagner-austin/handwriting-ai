from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol
from urllib.parse import urlparse

import handwriting_ai.jobs.digits as dj
from handwriting_ai.logging import init_logging
from handwriting_ai.training.mnist_train import TrainConfig, train_with_config

_DEFAULT_REDIS_ENV: Final[str] = "REDIS_URL"


@dataclass(frozen=True)
class _Args:
    payload_file: Path


def _parse_args(argv: list[str]) -> _Args:
    ap = argparse.ArgumentParser(description="Run a digits.train.v1 job payload once")
    ap.add_argument(
        "--payload-file",
        required=True,
        type=Path,
        help="Path to JSON payload for digits.train.v1",
    )
    ns = ap.parse_args(argv)
    # Avoid Any from argparse.Namespace by coercing through str -> Path
    return _Args(payload_file=Path(str(ns.payload_file)))


if TYPE_CHECKING:

    class _RedisPubProto(Protocol):  # pragma: no cover - typing only
        def publish(self, channel: str, message: str) -> int: ...

    def _redis_from_url(url: str) -> _RedisPubProto: ...
else:  # pragma: no cover - runtime import only

    def _redis_from_url(url: str):
        import redis

        return redis.Redis.from_url(url)


class _RedisPublisher:
    def __init__(self, url: str) -> None:
        self._url = url

    def publish(self, channel: str, message: str) -> int:
        # Return number of clients that received the message
        try:
            client = _redis_from_url(self._url)
            out_val = int(client.publish(channel, message))
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError):
            logging.getLogger("handwriting_ai").info("redis_publish_failed")
            return 0
        return out_val


def _make_publisher_from_env() -> dj.Publisher | None:
    url = os.getenv(_DEFAULT_REDIS_ENV)
    if not url or url.strip() == "":
        return None
    try:
        return _RedisPublisher(url)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        # If redis is not available or connection fails, fall back to no-op
        logging.getLogger("handwriting_ai").info("redis_unavailable_fallback_noop")
        return None


def _real_run_training(_: TrainConfig) -> Path:
    # Import inside function to avoid test-time dependency and speed
    from torchvision.datasets import MNIST

    cfg = _
    data_root = cfg.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    # MNIST returns PIL Image and int labels; matches MNISTLike protocol expectations
    train_base = MNIST(root=data_root.as_posix(), train=True, download=True)
    test_base = MNIST(root=data_root.as_posix(), train=False, download=True)
    out_dir = train_with_config(cfg, (train_base, test_base))
    _maybe_upload_artifacts(out_dir, cfg.model_id)
    return out_dir


def _get_env_str(name: str) -> str | None:
    v = os.getenv(name)
    return v if v is not None and v.strip() != "" else None


def _env_bool(name: str, default: bool = False) -> bool:
    v = _get_env_str(name)
    if v is None:
        return default
    s = v.strip().lower()
    return s in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    v = _get_env_str(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _encode_multipart(
    data: dict[str, str], files: dict[str, tuple[str, bytes, str]]
) -> tuple[bytes, str]:
    boundary = f"----hwai{os.getpid()}"
    crlf = "\r\n"
    body: list[bytes] = []
    for k, v in data.items():
        body.append((f"--{boundary}{crlf}").encode())
        body.append((f'Content-Disposition: form-data; name="{k}"{crlf}{crlf}').encode())
        body.append(v.encode("utf-8"))
        body.append(crlf.encode("utf-8"))
    for k, (filename, content, ctype) in files.items():
        body.append((f"--{boundary}{crlf}").encode())
        disp = f'Content-Disposition: form-data; name="{k}"; filename="{filename}"'
        body.append((disp + crlf).encode("utf-8"))
        body.append((f"Content-Type: {ctype}{crlf}{crlf}").encode())
        body.append(content)
        body.append(crlf.encode("utf-8"))
    body.append((f"--{boundary}--{crlf}").encode())
    payload = b"".join(body)
    return payload, f"multipart/form-data; boundary={boundary}"


def _http_post_multipart(
    url: str, body: bytes, content_type: str, headers_extra: dict[str, str], timeout_s: float
) -> tuple[int, bytes]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("invalid upload url")
    host = parsed.hostname or ""
    port = parsed.port
    conn = (
        HTTPSConnection(host, port=port, timeout=timeout_s)
        if parsed.scheme == "https"
        else HTTPConnection(host, port=port, timeout=timeout_s)
    )
    try:
        path = parsed.path or "/"
        if parsed.query:
            path = path + "?" + parsed.query
        headers = {"Content-Type": content_type, **headers_extra}
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        status = int(resp.status)
        data = resp.read()
        return status, data
    finally:
        from contextlib import suppress

        with suppress(OSError):
            conn.close()


def _maybe_upload_artifacts(model_dir: Path, model_id: str) -> None:
    url = _resolve_upload_url()
    api_key = _get_env_str("HANDWRITING_API_KEY")
    if api_key is None:
        raise RuntimeError("HANDWRITING_API_KEY is required")
    # Upload behavior knobs (read only shared globals, no alternate names)
    activate = True  # always request activation; can add a shared toggle later if needed
    retries = _env_int("HANDWRITING_API_MAX_RETRIES", 3)
    timeout_s = float(_env_int("HANDWRITING_API_TIMEOUT_SECONDS", 30))
    backoff_ms = 500  # fixed default; no env fallback by design
    strict = True  # strict by default; no env fallback by design

    manifest_bytes = (model_dir / "manifest.json").read_bytes()
    model_bytes = (model_dir / "model.pt").read_bytes()

    data: dict[str, str] = {"model_id": model_id, "activate": "true" if activate else "false"}
    files = {
        "manifest": ("manifest.json", manifest_bytes, "application/json"),
        "model": ("model.pt", model_bytes, "application/octet-stream"),
    }
    body, ct = _encode_multipart(data, files)
    headers: dict[str, str] = {"X-Api-Key": api_key} if api_key else {}

    attempt = 0
    while True:
        attempt += 1
        try:
            status, resp = _http_post_multipart(url, body, ct, headers, timeout_s)
        except (OSError, RuntimeError, ValueError, TypeError):
            status = 0
            resp = b""
        if status == 200:
            logging.getLogger("handwriting_ai").info(
                f"worker_upload_success status={status} bytes={len(resp)}"
            )
            return
        if attempt >= max(1, retries):
            logging.getLogger("handwriting_ai").info(
                f"worker_upload_failed status={status} attempts={attempt}"
            )
            if strict:
                raise RuntimeError("artifact upload failed")
            return
        # Backoff then retry
        import time as _t

        _t.sleep(float(backoff_ms) / 1000.0)


def _resolve_upload_url() -> str:
    base = _get_env_str("HANDWRITING_API_URL")
    if base is None:
        raise RuntimeError("HANDWRITING_API_URL is required")
    b = base.rstrip("/")
    return f"{b}/v1/admin/models/upload"


# No multi-name env readers by design (shared globals only)


def _run_once(payload_path: Path) -> int:
    raw = payload_path.read_text(encoding="utf-8")
    try:
        obj: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        logging.getLogger("handwriting_ai").error("invalid_payload_json")
        raise SystemExit(2) from exc
    if not isinstance(obj, dict):
        logging.getLogger("handwriting_ai").error("payload_not_object")
        return 2

    # Wire runtime implementations
    dj._run_training = _real_run_training
    dj._make_publisher = _make_publisher_from_env

    try:
        dj.process_train_job(obj)
    except (OSError, RuntimeError, ValueError, TypeError):
        # process_train_job is responsible for publishing failed; convert to non-zero exit
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _parse_args(argv or sys.argv[1:])
    return _run_once(args.payload_file)


if __name__ == "__main__":
    raise SystemExit(main())
