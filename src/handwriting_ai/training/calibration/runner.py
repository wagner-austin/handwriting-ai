from __future__ import annotations

import gc as _gc
import logging as _logging
import multiprocessing as _mp
import os as _os
import time as _time
from contextlib import suppress
from dataclasses import dataclass
from logging.handlers import QueueHandler as _QueueHandler
from logging.handlers import QueueListener as _QueueListener
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image as _Image
from torch.utils.data import Dataset as _TorchDataset

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.measure import CalibrationResult, _measure_candidate
from handwriting_ai.training.dataset import PreprocessDataset
from handwriting_ai.training.safety import MemoryGuardConfig, set_memory_guard_config


@dataclass(frozen=True)
class CandidateError:
    kind: str  # "timeout" | "oom" | "runtime"
    message: str
    exit_code: int | None


@dataclass(frozen=True)
class CandidateOutcome:
    ok: bool
    res: CalibrationResult | None
    error: CandidateError | None


@dataclass(frozen=True)
class BudgetConfig:
    start_pct_max: float
    abort_pct: float
    timeout_s: float
    max_failures: int


class CandidateRunner(Protocol):
    def run(  # pragma: no cover - signature line not counted reliably by coverage on some setups
        self,
        ds: PreprocessDataset | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome: ...


def _encode_result_kv(res: CalibrationResult) -> list[str]:
    """Encode a CalibrationResult into key=value lines for IPC."""
    inter_str = "" if res.interop_threads is None else str(int(res.interop_threads))
    return [
        "ok=1",
        f"intra_threads={int(res.intra_threads)}",
        f"interop_threads={inter_str}",
        f"num_workers={int(res.num_workers)}",
        f"batch_size={int(res.batch_size)}",
        f"samples_per_sec={float(res.samples_per_sec)}",
        f"p95_ms={float(res.p95_ms)}",
    ]


def _write_kv(out_path: str, lines: list[str]) -> None:
    """Write key=value lines atomically and durably.

    Writes to a temporary file, flushes and fsyncs content, then replaces the
    destination file. Attempts a directory fsync best-effort to reduce the
    chance of metadata lag on some filesystems.
    """
    tmp_path = f"{out_path}.tmp"
    content = "\n".join(lines)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(tmp_path, out_path)
    # Note: directory fsync intentionally omitted. File fsync above is
    # sufficient for our IPC durability requirements across platforms.


def _emit_result_file(out_path: str, res: CalibrationResult) -> None:
    """Encode and write a result file for IPC."""
    _write_kv(out_path, _encode_result_kv(res))


def _child_entry(
    out_path: str,
    spec: PreprocessSpec,
    cand: Candidate,
    samples: int,
    abort_pct: float,
    log_q: _mp.Queue[_logging.LogRecord],
) -> None:
    import time as _time

    from handwriting_ai.logging import init_logging

    # Child process needs to initialize its own logging
    init_logging()
    log = _logging.getLogger("handwriting_ai")
    # Remove any pre-existing StreamHandlers to avoid duplicate emission
    for h in tuple(log.handlers):
        if isinstance(h, _logging.StreamHandler):
            log.removeHandler(h)
    log.propagate = False

    # Bridge child logs to parent via a handler that is guaranteed to
    # implement logging.Handler. If a test stubs _QueueHandler with a minimal
    # object, adapt it to a proper handler to avoid runtime errors.

    class _ForwardingQueueHandler(_logging.Handler):
        """Handler that forwards records to a multiprocessing Queue."""

        def __init__(self, q: _mp.Queue[_logging.LogRecord]) -> None:
            super().__init__()
            self._q = q

        def emit(self, record: _logging.LogRecord) -> None:  # pragma: no cover - trivial
            # Best-effort forwarding; never raise from logging path
            with suppress(Exception):
                self._q.put(record)

    raw_handler = _QueueHandler(log_q)
    if isinstance(raw_handler, _logging.Handler):
        q_handler: _logging.Handler = raw_handler
    else:
        q_handler = _ForwardingQueueHandler(log_q)
    q_handler.setLevel(log.level)
    log.addHandler(q_handler)
    start_entry = _time.perf_counter()

    # Log immediately to prove we got here
    log.info("calibration_child_entry_start pid=%d", _os.getpid())

    log.info(
        "calibration_child_started pid=%d threads=%d workers=%d bs=%d base_kind=%s",
        _os.getpid(),
        cand.intra_threads,
        cand.num_workers,
        cand.batch_size,
        spec.base_kind,
    )
    try:
        # Rebuild dataset from spec to avoid pickling large objects
        log.info("calibration_child_building_dataset base_kind=%s", spec.base_kind)
        start_build = _time.perf_counter()
        ds = _build_dataset_from_spec(spec)
        build_elapsed = _time.perf_counter() - start_build
        log.info("calibration_child_dataset_built elapsed_ms=%.1f", build_elapsed * 1000)
        # Configure memory guard inside the child for calibration attempts.
        start_guard = _time.perf_counter()
        set_memory_guard_config(
            MemoryGuardConfig(
                enabled=True,
                threshold_percent=float(abort_pct),
                required_consecutive=3,
            )
        )
        guard_elapsed = _time.perf_counter() - start_guard
        log.info("calibration_child_guard_set elapsed_ms=%.1f", guard_elapsed * 1000)

        # Stream best-so-far so the parent can pick up a viable result early
        def _on_improvement(r: CalibrationResult) -> None:
            _emit_result_file(out_path, r)  # pragma: no cover (subprocess)

        start_measure = _time.perf_counter()
        res = _measure_candidate(ds, cand, samples, on_improvement=_on_improvement)
        measure_elapsed = _time.perf_counter() - start_measure
        log.info("calibration_child_measure_complete elapsed_s=%.1f", measure_elapsed)

        # Manual KV encoding of result (executes in child process)
        _emit_result_file(out_path, res)  # pragma: no cover (subprocess)

        total_elapsed = _time.perf_counter() - start_entry
        log.info("calibration_child_complete total_s=%.1f", total_elapsed)
    finally:
        # Flush QueueHandler to ensure all log records reach parent before exit
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()
        # Ensure prompt teardown
        _gc.collect()


class SubprocessRunner:
    def __init__(self) -> None:
        # Use spawn for consistent behavior across OSes
        self._ctx = _mp.get_context("spawn")

    def run(
        self,
        ds: PreprocessDataset | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        import tempfile as _tmp

        log = _logging.getLogger("handwriting_ai")
        out_dir = _tmp.mkdtemp(prefix="calib_child_")
        out_path = _os.path.join(out_dir, "result.txt")

        # Always pass a lightweight spec to the child
        spec = _to_spec(ds)

        spawn_start = _time.perf_counter()
        log_q: _mp.Queue[_logging.LogRecord] = self._ctx.Queue()
        # Mirror child logs to both root and application logger handlers (no fallbacks)
        _root_handlers = list(_logging.getLogger().handlers)
        _app_handlers = list(_logging.getLogger("handwriting_ai").handlers)
        _parent_handlers = tuple(_root_handlers + _app_handlers)
        listener = _QueueListener(log_q, *_parent_handlers, respect_handler_level=True)
        listener.start()
        proc = self._ctx.Process(
            target=_child_entry,
            args=(out_path, spec, cand, int(samples), float(budget.abort_pct), log_q),
        )
        # Ensure non-daemonic to allow DataLoader workers in child
        from contextlib import suppress as _suppress

        with _suppress(Exception):
            proc.daemon = False
        start = _time.perf_counter()
        proc.start()
        spawn_elapsed = _time.perf_counter() - spawn_start
        log.info(
            "calibration_parent_spawned threads=%d workers=%d bs=%d spawn_ms=%.1f timeout_s=%.1f",
            cand.intra_threads,
            cand.num_workers,
            cand.batch_size,
            spawn_elapsed * 1000,
            float(budget.timeout_s),
        )

        try:
            return self._wait_for_outcome(proc, out_path, start, float(budget.timeout_s))
        finally:
            if proc.is_alive():
                with suppress(Exception):
                    proc.kill()
                with suppress(Exception):
                    proc.join(1.0)
            with suppress(Exception):
                listener.stop()
            _gc.collect()

    def _wait_for_outcome(
        self,
        proc: BaseProcess,
        out_path: str,
        start: float,
        timeout_s: float,
    ) -> CandidateOutcome:
        # Poll for child result (file) with incremental waits
        while proc.is_alive():
            outcome = self._try_read_result(out_path, exited=False, exit_code=None)
            if outcome is not None:
                # Result file found, but child still alive - try to join briefly to
                # encourage clean exit and log flush; suppress on non-started mocks.
                remaining_time = timeout_s - (_time.perf_counter() - start)
                join_timeout = min(5.0, max(0.1, remaining_time))
                with suppress(Exception):
                    proc.join(timeout=join_timeout)
                return outcome
            if (_time.perf_counter() - start) >= timeout_s:
                # Timeout: terminate and mark
                with suppress(Exception):
                    proc.terminate()
                with suppress(Exception):
                    proc.join(2.0)
                return CandidateOutcome(
                    ok=False,
                    res=None,
                    error=CandidateError(
                        kind="timeout",
                        message="candidate timed out",
                        exit_code=None,
                    ),
                )
            _time.sleep(0.01)

        # Process not alive: try to read any pending outcome
        outcome2 = self._try_read_result(out_path, exited=True, exit_code=proc.exitcode)
        if outcome2 is not None:
            return outcome2

        # Determine exit condition
        code = proc.exitcode
        if code in (-9, 137):
            return CandidateOutcome(
                ok=False,
                res=None,
                error=CandidateError(
                    kind="oom", message="child killed (possible OOM)", exit_code=code
                ),
            )
        return CandidateOutcome(
            ok=False,
            res=None,
            error=CandidateError(
                kind="runtime", message=f"child exited code={code}", exit_code=code
            ),
        )

    # Queue forwarding handled by QueueListener

    @staticmethod
    def _try_read_result(
        out_path: str, *, exited: bool, exit_code: int | None
    ) -> CandidateOutcome | None:
        if not _os.path.exists(out_path):
            return None
        if not (_os.path.exists(out_path) and _os.access(out_path, _os.R_OK)):
            return None
        with open(out_path, encoding="utf-8") as f:
            content = f.read()
        # Parse simple key=value lines
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        data: dict[str, str] = {}
        for ln in lines:
            if "=" not in ln:
                return None
            k, v = ln.split("=", 1)
            data[k] = v

        if data.get("ok") == "1":
            # Required fields
            intra = int(data.get("intra_threads", "0"))
            inter_raw = data.get("interop_threads", "")
            inter = int(inter_raw) if inter_raw != "" else None
            nworkers = int(data.get("num_workers", "0"))
            bs = int(data.get("batch_size", "1"))
            sps = float(data.get("samples_per_sec", "0.0"))
            p95 = float(data.get("p95_ms", "0.0"))
            res = CalibrationResult(
                intra_threads=intra,
                interop_threads=inter,
                num_workers=nworkers,
                batch_size=bs,
                samples_per_sec=sps,
                p95_ms=p95,
            )
            return CandidateOutcome(ok=True, res=res, error=None)

        if data.get("ok") == "0":
            msg = data.get("error_message", "")
            return CandidateOutcome(
                ok=False,
                res=None,
                error=CandidateError(
                    kind="runtime", message=msg, exit_code=exit_code if exited else None
                ),
            )
        return None


__all__ = [
    "BudgetConfig",
    "CandidateError",
    "CandidateOutcome",
    "CandidateRunner",
    "SubprocessRunner",
]

# ---------- Helpers (module-internal) ----------


@runtime_checkable
class _KnobsProto(Protocol):
    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str


# Module-level dataset classes for Windows spawn pickle compatibility


class _InlineDataset(_TorchDataset[tuple[_Image.Image, int]]):
    """Inline synthetic dataset for calibration."""

    def __init__(self, n: int, sleep_s: float, fail: bool) -> None:
        self._n = int(n)
        self._sleep = float(sleep_s)
        self._fail = bool(fail)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[_Image.Image, int]:
        import time as _time

        if self._fail:
            raise RuntimeError("fail-item")
        if self._sleep > 0:
            _time.sleep(self._sleep)
        return _Image.new("L", (28, 28), color=0), int(idx % 10)


class _MNISTRawDataset(_TorchDataset[tuple[_Image.Image, int]]):
    """MNIST dataset from raw bytes (for calibration)."""

    def __init__(self, images: list[bytes], labels: list[int]) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[_Image.Image, int]:
        i = int(idx)
        img = _Image.frombytes("L", (28, 28), self._images[i])
        return img, self._labels[i]


def _to_spec(ds: PreprocessDataset | PreprocessSpec) -> PreprocessSpec:
    if isinstance(ds, PreprocessSpec):
        return ds
    k = ds.knobs
    aug = AugmentSpec(
        augment=bool(k.enable),
        aug_rotate=float(k.rotate_deg),
        aug_translate=float(k.translate_frac),
        noise_prob=float(k.noise_prob),
        noise_salt_vs_pepper=float(k.noise_salt_vs_pepper),
        dots_prob=float(k.dots_prob),
        dots_count=int(k.dots_count),
        dots_size_px=int(k.dots_size_px),
        blur_sigma=float(k.blur_sigma),
        morph=str(k.morph_mode),
    )
    inline = InlineSpec(n=len(ds), sleep_s=0.0, fail=False)
    return PreprocessSpec(base_kind="inline", mnist=None, inline=inline, augment=aug)


def _build_dataset_from_spec(spec: PreprocessSpec) -> PreprocessDataset:
    class _Cfg:
        augment = bool(spec.augment.augment)
        aug_rotate = float(spec.augment.aug_rotate)
        aug_translate = float(spec.augment.aug_translate)
        noise_prob = float(spec.augment.noise_prob)
        noise_salt_vs_pepper = float(spec.augment.noise_salt_vs_pepper)
        dots_prob = float(spec.augment.dots_prob)
        dots_count = int(spec.augment.dots_count)
        dots_size_px = int(spec.augment.dots_size_px)
        blur_sigma = float(spec.augment.blur_sigma)
        morph = str(spec.augment.morph)
        morph_kernel_px = 1
        batch_size = 1

    if spec.base_kind == "mnist":
        return _build_mnist_dataset(spec)

    if spec.base_kind == "inline":
        if spec.inline is None:
            raise RuntimeError("inline spec missing details")

        base = _InlineDataset(spec.inline.n, spec.inline.sleep_s, spec.inline.fail)
        return PreprocessDataset(base, _Cfg())

    raise RuntimeError(f"unknown base_kind: {spec.base_kind}")


def _mnist_find_raw_dir(root: Path) -> Path:
    p = root / "MNIST" / "raw"
    return p if p.exists() else root


def _mnist_read_images_labels(root: Path, train: bool) -> tuple[list[bytes], list[int]]:
    import gzip as _gzip

    rd = _mnist_find_raw_dir(root)
    pref = "train" if train else "t10k"
    img_path = rd / f"{pref}-images-idx3-ubyte.gz"
    lbl_path = rd / f"{pref}-labels-idx1-ubyte.gz"
    if not (img_path.exists() and lbl_path.exists()):
        raise RuntimeError("MNIST raw files not found under root")
    with _gzip.open(img_path, "rb") as fimg:
        header = fimg.read(16)
        if len(header) != 16:
            raise RuntimeError("invalid MNIST images header")
        magic = int.from_bytes(header[0:4], "big")
        n = int.from_bytes(header[4:8], "big")
        rows = int.from_bytes(header[8:12], "big")
        cols = int.from_bytes(header[12:16], "big")
        if magic != 2051 or rows != 28 or cols != 28:
            raise RuntimeError("invalid MNIST images file")
        total = int(n * rows * cols)
        data = fimg.read(total)
        if len(data) != total:
            raise RuntimeError("truncated MNIST images file")
    with _gzip.open(lbl_path, "rb") as flbl:
        header2 = flbl.read(8)
        if len(header2) != 8:
            raise RuntimeError("invalid MNIST labels header")
        magic2 = int.from_bytes(header2[0:4], "big")
        n2 = int.from_bytes(header2[4:8], "big")
        if magic2 != 2049 or n2 != n:
            raise RuntimeError("invalid MNIST labels file")
        labels_raw = flbl.read(int(n2))
        if len(labels_raw) != int(n2):
            raise RuntimeError("truncated MNIST labels file")
    stride = 28 * 28
    imgs = [data[i * stride : (i + 1) * stride] for i in range(int(n))]
    labels = [int(b) for b in labels_raw]
    return imgs, labels


def _build_mnist_dataset(spec: PreprocessSpec) -> PreprocessDataset:
    import logging as _logging

    log = _logging.getLogger("handwriting_ai")

    log.info("_build_mnist_dataset_start root=%s", spec.mnist.root if spec.mnist else None)

    if spec.mnist is None:
        raise RuntimeError("mnist spec missing details")

    log.info("_build_mnist_dataset_reading_files")
    import time as _time

    start_read = _time.perf_counter()
    imgs, labels = _mnist_read_images_labels(spec.mnist.root, bool(spec.mnist.train))
    read_elapsed_ms = (_time.perf_counter() - start_read) * 1000
    log.info("_build_mnist_dataset_files_read count=%d elapsed_ms=%.1f", len(imgs), read_elapsed_ms)

    class _Cfg:
        augment = bool(spec.augment.augment)
        aug_rotate = float(spec.augment.aug_rotate)
        aug_translate = float(spec.augment.aug_translate)
        noise_prob = float(spec.augment.noise_prob)
        noise_salt_vs_pepper = float(spec.augment.noise_salt_vs_pepper)
        dots_prob = float(spec.augment.dots_prob)
        dots_count = int(spec.augment.dots_count)
        dots_size_px = int(spec.augment.dots_size_px)
        blur_sigma = float(spec.augment.blur_sigma)
        morph = str(spec.augment.morph)
        morph_kernel_px = 1
        batch_size = 1

    return PreprocessDataset(_MNISTRawDataset(imgs, labels), _Cfg())
