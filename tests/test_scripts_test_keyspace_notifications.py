from __future__ import annotations

import asyncio

import pytest
from scripts.test_keyspace_notifications import test_keyspace_notifications as _demo


def test_keyspace_notifications_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    with pytest.raises(SystemExit):
        asyncio.run(_demo())
