"""Coverage test for interrupted event in digits.py to achieve 100%."""

from __future__ import annotations

from handwriting_ai.events import digits as ev


def test_interrupted_event_structure() -> None:
    """Test line 375: interrupted() function returns correct structure."""
    ctx = ev.Context(
        request_id="req-123",
        user_id=42,
        model_id="model-abc",
        run_id="run-xyz",
    )

    result = ev.interrupted(ctx, path="/path/to/artifacts")

    assert result["type"] == "digits.train.interrupted.v1"
    assert result["request_id"] == "req-123"
    assert result["user_id"] == 42
    assert result["model_id"] == "model-abc"
    assert result["run_id"] == "run-xyz"
    assert result["path"] == "/path/to/artifacts"
    assert "ts" in result
