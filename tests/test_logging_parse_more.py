from __future__ import annotations

from handwriting_ai.logging import _is_float_str, _parse_evt_fields


def test_parse_evt_fields_handles_tokens_without_equals_and_empty_key() -> None:
    msg = "EVT event=done junk bar=1 =2 confidence="
    out = _parse_evt_fields(msg)
    # 'junk' ignored (no '='); '=2' ignored (empty key);
    # confidence= with empty value should not convert to float
    assert out.get("event") == "done"
    # bar is not a recognized numeric field; remains string
    assert out.get("bar") == "1"
    assert "" not in out
    assert out.get("confidence") == ""


def test_is_float_str_handles_empty_string() -> None:
    assert _is_float_str("") is False
