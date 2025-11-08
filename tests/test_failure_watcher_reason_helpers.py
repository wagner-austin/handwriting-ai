from __future__ import annotations

import handwriting_ai.jobs.failure_watcher as mod


def test_coerce_str_variants() -> None:
    # Directly exercise helper branches: str, bytes, and unsupported type
    assert mod._coerce_str("x") == "x"
    assert mod._coerce_str(b"y") == "y"

    class _T:
        pass

    assert mod._coerce_str(_T()) is None


def test_detect_failed_reason_branches() -> None:
    # Not implementing hget -> returns None (unsupported client)
    class _NoHash:
        def zrange(self, key: str, start: int, end: int) -> list[str]:  # pragma: no cover
            return []

    assert mod._detect_failed_reason(_NoHash(), "jid") is None

    # hget raises -> except branch; then no usable value -> None
    class _Raises:
        def hget(self, key: str, field: str) -> str | bytes | None:
            raise RuntimeError("boom")

    assert mod._detect_failed_reason(_Raises(), "jid") is None

    # hget returns whitespace-only string -> treated as absent
    class _Blank:
        def hget(self, key: str, field: str) -> str | bytes | None:  # pragma: no cover
            return "  "

    assert mod._detect_failed_reason(_Blank(), "jid") is None
