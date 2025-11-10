from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from handwriting_ai.jobs.watcher.ports import RedisFactory
from handwriting_ai.jobs.watcher.publisher import RedisPublisher
from handwriting_ai.jobs.watcher.store import RedisProcessedStore

if TYPE_CHECKING:  # typing-only import for protocol compatibility
    from handwriting_ai.jobs.watcher.adapters import (
        RedisClient as _RedisClient,
    )


@dataclass
class _Client:
    last: tuple[str, str] | None = None
    to_raise_publish: Exception | None = None
    to_raise_seen: Exception | None = None
    to_raise_mark: Exception | None = None
    seen_set: set[str] = field(default_factory=set)

    def publish(self, channel: str, message: str) -> int:
        if self.to_raise_publish is not None:
            raise self.to_raise_publish
        self.last = (channel, message)
        return 1

    def sismember(self, key: str, member: str) -> bool:
        if self.to_raise_seen is not None:
            raise self.to_raise_seen
        return member in self.seen_set

    def sadd(self, key: str, *members: str) -> int:
        if self.to_raise_mark is not None:
            raise self.to_raise_mark
        for m in members:
            self.seen_set.add(m)
        return len(members)


def _factory(client: _Client) -> RedisFactory:
    def _f(url: str, *, decode_responses: bool = False) -> _RedisClient:
        return client

    return _f


def test_redis_publisher_success() -> None:
    client = _Client()
    pub = RedisPublisher("redis://local/0", redis_factory=_factory(client))
    out = pub.publish("ch", "msg")
    assert out == 1 and client.last == ("ch", "msg")


def test_redis_publisher_error_raised() -> None:
    client = _Client(to_raise_publish=OSError("boom"))
    pub = RedisPublisher("redis://local/0", redis_factory=_factory(client))
    with pytest.raises(OSError):
        pub.publish("ch", "msg")


def test_store_seen_and_mark_success() -> None:
    client = _Client()
    st = RedisProcessedStore("redis://local/0", key="k", redis_factory=_factory(client))
    assert st.seen("a") is False
    st.mark("a")
    assert st.seen("a") is True


def test_store_seen_and_mark_errors() -> None:
    client = _Client(to_raise_seen=OSError("x"), to_raise_mark=OSError("y"))
    st = RedisProcessedStore("redis://local/0", key="k", redis_factory=_factory(client))
    with pytest.raises(OSError):
        st.seen("a")
    # Clear seen raise to exercise mark raise
    client.to_raise_seen = None
    with pytest.raises(OSError):
        st.mark("a")
