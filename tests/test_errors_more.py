from __future__ import annotations

from handwriting_ai.errors import ErrorCode, status_for


def test_status_for_preprocessing_failed_and_default() -> None:
    assert int(status_for(ErrorCode.preprocessing_failed)) == 400
    # Default branch hit via internal_error mapping (no explicit case → 500)
    assert int(status_for(ErrorCode.internal_error)) == 500
