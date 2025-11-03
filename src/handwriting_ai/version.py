from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VersionInfo:
    service: str
    version: str
    build: str | None
    commit: str | None


def get_version() -> VersionInfo:
    version_str = _pkg_version()
    return VersionInfo(
        service="handwriting-ai",
        version=version_str,
        build=os.getenv("BUILD_ID"),
        commit=os.getenv("GIT_COMMIT") or os.getenv("COMMIT_SHA"),
    )


def _pkg_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("handwriting-ai")
    except PackageNotFoundError:
        from .logging import get_logger

        logger = get_logger()
        logger.info("pkg_version_fallback")
        return "0.1.0"
