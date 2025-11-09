from __future__ import annotations

import pytest

import handwriting_ai.jobs.watcher.logic as logic


def test_coerce_str_variants() -> None:
    # Directly exercise helper branches: str, bytes, and unsupported type
    assert logic.coerce_str("x") == "x"
    assert logic.coerce_str(b"y") == "y"

    class _T:
        pass

    assert logic.coerce_str(_T()) is None


def test_detect_failed_reason_branches() -> None:
    # Not implementing hget -> returns None (unsupported client)
    class _NoHash:
        def zrange(self, key: str, start: int, end: int) -> list[str]:  # pragma: no cover
            return []

    assert logic.detect_failed_reason(_NoHash(), "jid") is None

    # hget raises -> now we raise from helper
    class _Raises:
        def hget(self, key: str, field: str) -> str | bytes | None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        logic.detect_failed_reason(_Raises(), "jid")

    # hget returns whitespace-only string -> treated as absent
    class _Blank:
        def hget(self, key: str, field: str) -> str | bytes | None:  # pragma: no cover
            return "  "

    assert logic.detect_failed_reason(_Blank(), "jid") is None


def test_coerce_job_ids_ignores_other_types() -> None:
    out = logic.coerce_job_ids([123, object(), b"", "ok"])
    assert out == ["ok"]
