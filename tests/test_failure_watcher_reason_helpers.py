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

    # hget returns non-empty string -> returned directly
    class _Has:
        def hget(self, key: str, field: str) -> str | bytes | None:  # pragma: no cover
            return "reason"

    assert logic.detect_failed_reason(_Has(), "jid") == "reason"


def test_coerce_job_ids_ignores_other_types() -> None:
    out = logic.coerce_job_ids([123, object(), b"", "ok"])
    assert out == ["ok"]


def test_summarize_exc_info_default_and_extract_payload_empty() -> None:
    # Default branch when not a non-empty string
    assert logic.summarize_exc_info(None) == "job failed"

    class _Job:
        args: object = []

    # No args -> empty payload
    assert logic.extract_payload(_Job()) == {}


def test_make_logger_and_extract_payload_non_dict() -> None:
    # make_logger returns our module logger
    lg = logic.make_logger()
    assert lg.name == "handwriting_ai"

    class _Job:
        args: object = [42]

    # args present but first arg not dict -> {}
    assert logic.extract_payload(_Job()) == {}
