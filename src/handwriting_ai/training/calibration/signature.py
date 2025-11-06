from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

from handwriting_ai.training.resources import ResourceLimits


@dataclass(frozen=True)
class CalibrationSignature:
    cpu_cores: int
    mem_bytes: int | None
    os: str
    py: str
    torch: str


def make_signature(limits: ResourceLimits) -> CalibrationSignature:
    return CalibrationSignature(
        cpu_cores=int(limits.cpu_cores),
        mem_bytes=limits.memory_bytes,
        os=f"{platform.system()}-{platform.release()}",
        py=platform.python_version(),
        torch=str(torch.__version__),
    )
