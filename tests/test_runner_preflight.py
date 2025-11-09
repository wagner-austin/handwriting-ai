from __future__ import annotations

import sys
import types

import pytest


def test_preflight_raises_when_registries_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Provide a dummy rq.registry lacking required attributes
    mod_rq = types.ModuleType("rq")
    mod_reg = types.ModuleType("rq.registry")
    # No StartedJobRegistry / CanceledJobRegistry attributes on purpose
    mod_rq.__path__ = []
    monkeypatch.setitem(sys.modules, "rq", mod_rq)
    monkeypatch.setitem(sys.modules, "rq.registry", mod_reg)

    import importlib

    # Reload runner to ensure it picks up our patched module on import
    import handwriting_ai.jobs.watcher.runner as runner_mod

    importlib.reload(runner_mod)

    with pytest.raises(RuntimeError) as ei:
        runner_mod._preflight_registries()

    msg = str(ei.value)
    assert "StartedJobRegistry" in msg and "CanceledJobRegistry" in msg
