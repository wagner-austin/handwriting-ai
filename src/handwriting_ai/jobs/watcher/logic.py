from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


def make_logger() -> logging.Logger:
    return logging.getLogger("handwriting_ai")


def summarize_exc_info(exc_info: object) -> str:
    if isinstance(exc_info, str) and exc_info.strip():
        lines = exc_info.strip().splitlines()
        full_text = "\n".join(lines[:10])
        if "signal 9" in full_text.lower() or "sigkill" in full_text.lower():
            return (
                "OOM kill detected (signal 9 / SIGKILL) - "
                "worker terminated by system due to memory exhaustion"
            )
        if "waitpid returned 9" in full_text:
            return "Worker killed by OS (signal 9) - likely OOM (out of memory)"
        return lines[0][:300] if lines else "job failed"
    return "job failed"


def coerce_str(val: object) -> str | None:
    if isinstance(val, str):
        return val
    if isinstance(val, bytes | bytearray):
        return bytes(val).decode("utf-8", errors="ignore")
    return None


def extract_payload(job: object) -> dict[str, object]:
    args: object = getattr(job, "args", None)
    if isinstance(args, list | tuple) and len(args) > 0:
        first = args[0]
        if isinstance(first, dict):
            out: dict[str, object] = {}
            for k in ("request_id", "user_id", "model_id", "type"):
                out[k] = first.get(k)
            return out
    return {}


def coerce_job_ids(items: Sequence[object]) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, bytes):
            s = it.decode("utf-8", errors="ignore")
            if s:
                out.append(s)
    return out


@runtime_checkable
class _RedisHashProto(Protocol):  # pragma: no cover - typing only
    def hget(self, key: str, field: str) -> str | bytes | None: ...


def detect_failed_reason(conn: object, job_id: str) -> str | None:
    if not isinstance(conn, _RedisHashProto):
        return None
    key = f"rq:job:{job_id}"
    for field in ("failed_reason", "failure_reason", "exc_info"):
        try:
            raw = conn.hget(key, field)
        except (RuntimeError, ValueError, TypeError, OSError) as _e:
            logging.getLogger("handwriting_ai").error("redis_hget_error error=%s", str(_e))
            raise
        s = coerce_str(raw)
        if s and s.strip():
            return s
    return None
