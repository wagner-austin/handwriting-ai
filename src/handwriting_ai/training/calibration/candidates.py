from __future__ import annotations

from dataclasses import dataclass

import torch

from handwriting_ai.training.resources import ResourceLimits, compute_max_batch_size


@dataclass(frozen=True)
class Candidate:
    intra_threads: int
    interop_threads: int | None
    num_workers: int
    batch_size: int


def _candidate_threads(limits: ResourceLimits) -> list[int]:
    a = max(1, int(limits.cpu_cores // 2))
    b = max(1, int(limits.cpu_cores))
    return sorted({a, b})


def _candidate_workers(limits: ResourceLimits) -> list[int]:
    """Enumerate worker candidates within safe, bounded limits.

    We keep enumeration compact for Stage A calibration while respecting
    memory constraints. This does not pick a value heuristically; it bounds
    the search space that calibration measures empirically.
    """
    cores = max(0, int(limits.cpu_cores))
    if cores <= 1:
        return [0]
    # Base bound by cores to keep search compact
    upper_by_cores = min(2, cores)
    # Memory-aware bound: on sub-1.2 GiB environments, avoid exploring >1 worker
    upper_by_mem = upper_by_cores
    mem = limits.memory_bytes
    if mem is not None and mem < int(1.2 * 1024 * 1024 * 1024):
        upper_by_mem = min(1, upper_by_cores)
    upper = min(upper_by_cores, upper_by_mem)
    return list(range(0, upper + 1))


def _generate_candidates(limits: ResourceLimits, requested_batch_size: int) -> list[Candidate]:
    cap = compute_max_batch_size(limits.memory_bytes)
    base_bs = int(requested_batch_size)
    if cap is not None:
        base_bs = min(base_bs, int(cap))
    out: list[Candidate] = []
    for intra in _candidate_threads(limits):
        inter = max(1, intra // 2) if hasattr(torch, "set_num_interop_threads") else None
        for workers in _candidate_workers(limits):
            out.append(Candidate(intra, inter, workers, base_bs))
    return out
