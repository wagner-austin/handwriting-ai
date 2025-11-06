from __future__ import annotations

import importlib
import types

import pytest

import handwriting_ai.inference.engine as eng


def test_build_model_handles_missing_attrs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _TV:
        def resnet18(self, *, weights: object | None, num_classes: int) -> object:
            class _M:
                # Deliberately no conv1/maxpool attributes
                pass

            return _M()

    def _import_module(name: str) -> object:
        # Return a module-like object exposing resnet18
        m = types.SimpleNamespace()
        m.resnet18 = _TV().resnet18
        return m

    monkeypatch.setattr(importlib, "import_module", _import_module, raising=True)
    m = eng._build_model("resnet18", 10)
    # Ensure we get back the stub object without conv1/maxpool replacements
    assert not hasattr(m, "conv1") and not hasattr(m, "maxpool")
