from __future__ import annotations

import pytest
import torch

from handwriting_ai.training.mnist_train import _configure_interop_threads


def test_configure_interop_threads_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int | None] = {"v": None}

    def _ok(n: int) -> None:
        called["v"] = int(n)

    monkeypatch.setattr(torch, "set_num_interop_threads", _ok, raising=True)
    _configure_interop_threads(2)
    assert called["v"] == 2


def test_configure_interop_threads_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(n: int) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "set_num_interop_threads", _boom, raising=True)
    # Should raise after logging
    with pytest.raises(RuntimeError, match="boom"):
        _configure_interop_threads(1)


def test_configure_interop_threads_skips_on_none() -> None:
    # No-op when interop_threads is None
    _configure_interop_threads(None)


def test_configure_interop_threads_skips_without_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(torch, "set_num_interop_threads"):
        # monkeypatch will restore the attribute after the test automatically
        monkeypatch.delattr(torch, "set_num_interop_threads", raising=False)
    _configure_interop_threads(2)
