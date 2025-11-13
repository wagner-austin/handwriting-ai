from __future__ import annotations

import gzip
from pathlib import Path
from typing import Protocol

import pytest
from PIL import Image

from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    MNISTSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.runner import (
    _build_dataset_from_spec,
    _child_entry,
    _mnist_read_images_labels,
    _to_spec,
)
from handwriting_ai.training.dataset import PreprocessDataset


class _TinyBase:
    def __init__(self, n: int) -> None:
        self._n = int(n)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        return Image.new("L", (28, 28), color=0), int(idx % 10)


class _CfgProto(Protocol):
    augment: bool
    aug_rotate: float
    aug_translate: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph: str
    morph_kernel_px: int
    batch_size: int


class _Cfg:
    def __init__(self) -> None:
        self.augment = True
        self.aug_rotate = 5.0
        self.aug_translate = 0.1
        self.noise_prob = 0.2
        self.noise_salt_vs_pepper = 0.6
        self.dots_prob = 0.1
        self.dots_count = 2
        self.dots_size_px = 1
        self.blur_sigma = 0.5
        self.morph = "none"
        self.morph_kernel_px = 1
        self.batch_size = 1


def _write_gzip(path: Path, data: bytes) -> None:
    with gzip.open(path, "wb") as f:
        f.write(data)


def _mk_images_header(n: int, rows: int = 28, cols: int = 28, magic: int = 2051) -> bytes:
    return (
        magic.to_bytes(4, "big")
        + int(n).to_bytes(4, "big")
        + int(rows).to_bytes(4, "big")
        + int(cols).to_bytes(4, "big")
    )


def _mk_labels_header(n: int, magic: int = 2049) -> bytes:
    return magic.to_bytes(4, "big") + int(n).to_bytes(4, "big")


def test_to_spec_from_dataset() -> None:
    base = _TinyBase(5)
    ds = PreprocessDataset(base, _Cfg())
    spec = _to_spec(ds)
    assert isinstance(spec, PreprocessSpec)
    assert spec.base_kind == "inline"
    assert spec.inline is not None and spec.inline.n == len(ds)
    # Ensure knobs are mapped
    assert spec.augment.augment is True and spec.augment.aug_rotate == pytest.approx(5.0)


def test_to_spec_pass_through_for_spec() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=3, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    # Identity for spec input
    assert _to_spec(spec) is spec


def test_build_dataset_from_spec_inline_success() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=7, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds = _build_dataset_from_spec(spec)
    assert isinstance(ds, PreprocessDataset)
    assert len(ds) == 7


def test_build_dataset_from_spec_inline_fail_and_sleep(tmp_path: Path) -> None:
    # fail=True triggers RuntimeError path in _InlineDataset.__getitem__
    spec_fail = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=1, sleep_s=0.0, fail=True),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds_fail = _build_dataset_from_spec(spec_fail)
    with pytest.raises(RuntimeError, match="fail-item"):
        _ = ds_fail[0]

    # sleep_s>0 exercises sleep branch in _InlineDataset.__getitem__
    spec_sleep = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=1, sleep_s=0.001, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds_sleep = _build_dataset_from_spec(spec_sleep)
    x, y = ds_sleep[0]
    assert x.shape[-2:] == (28, 28) and int(y) in range(10)


def test_build_dataset_from_spec_inline_missing_details() -> None:
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=None,
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    with pytest.raises(RuntimeError, match="inline spec missing details"):
        _build_dataset_from_spec(spec)


def test_mnist_read_images_labels_happy_path(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 3
    img_header = _mk_images_header(n)
    lbl_header = _mk_labels_header(n)
    images = bytes([0] * (n * 28 * 28))
    labels = bytes([i % 10 for i in range(n)])
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + images)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", lbl_header + labels)
    imgs, lbls = _mnist_read_images_labels(tmp_path, True)
    assert len(imgs) == n and len(lbls) == n


def test_mnist_read_images_labels_missing_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="MNIST raw files not found"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_images_header(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", b"bad")  # < 16 bytes
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST images header"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_images_magic(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    img_header = _mk_images_header(1, rows=28, cols=28, magic=9999)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + bytes([0] * 784))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST images file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_truncated_images(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 2
    img_header = _mk_images_header(n)
    # Missing some bytes
    images = bytes([0] * (n * 28 * 28 - 1))
    _write_gzip(raw / "train-images-idx3-ubyte.gz", img_header + images)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n) + bytes([0, 1]))
    with pytest.raises(RuntimeError, match="truncated MNIST images file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_labels_header(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 1
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * 784))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", b"bad")
    with pytest.raises(RuntimeError, match="invalid MNIST labels header"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_bad_labels_magic(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 1
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * 784))
    # Wrong magic (2048)
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n, magic=2048) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_count_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    # n mismatch between image header and labels header (2 vs 1)
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(2) + bytes([0] * (2 * 784)))
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(1) + bytes([0]))
    with pytest.raises(RuntimeError, match="invalid MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_mnist_read_images_labels_truncated_labels(tmp_path: Path) -> None:
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 3
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * (n * 784)))
    # Only n-1 labels
    _write_gzip(raw / "train-labels-idx1-ubyte.gz", _mk_labels_header(n) + bytes([0] * (n - 1)))
    with pytest.raises(RuntimeError, match="truncated MNIST labels file"):
        _mnist_read_images_labels(tmp_path, True)


def test_build_dataset_from_spec_mnist(tmp_path: Path) -> None:
    # Create valid raw MNIST files
    raw = tmp_path / "MNIST" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = 4
    _write_gzip(raw / "train-images-idx3-ubyte.gz", _mk_images_header(n) + bytes([0] * (n * 784)))
    _write_gzip(
        raw / "train-labels-idx1-ubyte.gz",
        _mk_labels_header(n) + bytes([i % 10 for i in range(n)]),
    )

    spec = PreprocessSpec(
        base_kind="mnist",
        mnist=MNISTSpec(root=tmp_path, train=True),
        inline=None,
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    ds = _build_dataset_from_spec(spec)
    assert isinstance(ds, PreprocessDataset)
    assert len(ds) == n
    # Exercise _MNISTRawDataset.__getitem__ through wrapper
    x, y = ds[0]
    assert x.shape[-2:] == (28, 28) and int(y) in range(10)


def test_build_mnist_dataset_missing_details_raises() -> None:
    import handwriting_ai.training.calibration.runner as rmod

    spec = PreprocessSpec(
        base_kind="mnist",
        mnist=None,
        inline=None,
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    with pytest.raises(RuntimeError, match="mnist spec missing details"):
        rmod._build_mnist_dataset(spec)


def test_child_entry_inline_executes_and_writes_result(tmp_path: Path) -> None:
    # Build a minimal inline spec and candidate
    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=4, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )

    import logging
    import multiprocessing as mp
    from multiprocessing.queues import Queue as MPQueue

    from handwriting_ai.training.calibration.candidates import Candidate

    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=2)
    out_file = str(tmp_path / "child_result.txt")

    q: MPQueue[logging.LogRecord] = mp.get_context("spawn").Queue()
    # Run inline inside this process
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
    content = Path(out_file).read_text(encoding="utf-8")
    assert "ok=1" in content and "batch_size=2" in content


def test_run_finally_kills_alive_child(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Precreate expected outdir and result
    out_dir = tmp_path / "calib_child_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.txt"
    out_path.write_text(
        "ok=1\n"
        "intra_threads=1\n"
        "interop_threads=\n"
        "num_workers=0\n"
        "batch_size=1\n"
        "samples_per_sec=1.0\n"
        "p95_ms=1.0\n",
        encoding="utf-8",
    )

    # Stub mkdtemp to return our directory
    import tempfile as _tmp

    def _mk(prefix: str) -> str:
        return str(out_dir)

    monkeypatch.setattr(_tmp, "mkdtemp", _mk, raising=True)

    # Dummy context and process that stays alive until kill/join called
    class _Proc:
        def __init__(self) -> None:
            self._alive = True
            self._killed = False
            self._joined = False

        def start(self) -> None:  # no-op
            self._alive = True

        def is_alive(self) -> bool:
            return True

        def kill(self) -> None:
            self._killed = True

        def join(self, timeout: float | None = None) -> None:
            self._joined = True

        @property
        def exitcode(self) -> int:
            return 0

    class _Ctx:
        def Process(self, target: object, args: tuple[object, ...]) -> _Proc:  # noqa: N802
            return _Proc()

        class _Q:
            def put(self, _: object) -> None:
                return None

        def Queue(self) -> _Q:  # noqa: N802
            return _Ctx._Q()

    # Replace QueueListener with no-op to avoid threading
    import handwriting_ai.training.calibration.runner as rmod

    class _NoopListener:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    def _make_listener(*args: object, **kwargs: object) -> _NoopListener:
        return _NoopListener()

    monkeypatch.setattr(rmod, "_QueueListener", _make_listener, raising=True)

    runner = rmod.SubprocessRunner()
    # Monkeypatch context used by runner
    monkeypatch.setattr(runner, "_ctx", _Ctx(), raising=True)

    from handwriting_ai.training.calibration.candidates import Candidate

    ds = PreprocessDataset(_TinyBase(2), _Cfg())
    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=1)
    budget = rmod.BudgetConfig(start_pct_max=99.0, abort_pct=99.0, timeout_s=10.0, max_failures=1)
    out = runner.run(ds, cand, samples=1, budget=budget)
    assert out.ok and out.res is not None and int(out.res.batch_size) == 1


def test_child_entry_flush_branch_no_flush_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import logging
    import multiprocessing as mp
    from multiprocessing.queues import Queue as MPQueue

    import handwriting_ai.training.calibration.runner as rmod

    class _QH(logging.Handler):
        """Minimal queue handler that appears to lack a usable ``flush``.

        Subclasses logging.Handler to keep types strict while exercising the
        branch that checks for a flush attribute without providing one that
        can be called successfully.
        """

        def __init__(self, q: MPQueue[logging.LogRecord]) -> None:
            super().__init__()
            self.q = q

        def emit(self, record: logging.LogRecord) -> None:
            """Emit a record by putting it in the queue."""
            self.q.put_nowait(record)

        def __getattr__(self, name: str) -> object:
            """Pretend that 'flush' does not exist for hasattr checks."""
            if name == "flush":
                raise AttributeError(f"'{type(self).__name__}' object has no attribute 'flush'")
            raise AttributeError(name)

    monkeypatch.setattr(rmod, "_QueueHandler", _QH, raising=True)

    spec = PreprocessSpec(
        base_kind="inline",
        mnist=None,
        inline=InlineSpec(n=1, sleep_s=0.0, fail=False),
        augment=AugmentSpec(
            augment=False,
            aug_rotate=0.0,
            aug_translate=0.0,
            noise_prob=0.0,
            noise_salt_vs_pepper=0.5,
            dots_prob=0.0,
            dots_count=0,
            dots_size_px=1,
            blur_sigma=0.0,
            morph="none",
        ),
    )
    out_file = str(tmp_path / "child_nf.txt")
    from handwriting_ai.training.calibration.candidates import Candidate

    cand = Candidate(intra_threads=1, interop_threads=None, num_workers=0, batch_size=1)

    q: MPQueue[logging.LogRecord] = mp.get_context("spawn").Queue()
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
    # Cleanup: remove our stub handler if it was attached
    app_log = logging.getLogger("handwriting_ai")
    for h in list(app_log.handlers):
        if h.__class__.__name__ == _QH.__name__:
            app_log.removeHandler(h)
    assert Path(out_file).exists()
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
