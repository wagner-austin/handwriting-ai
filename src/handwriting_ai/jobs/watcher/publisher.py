from __future__ import annotations

import logging
from typing import Protocol

from .adapters import redis_from_url as _default_redis_from_url
from .ports import RedisFactory


class Publisher(Protocol):
    def publish(self, channel: str, message: str) -> int: ...


class RedisPublisher:
    def __init__(self, url: str, *, redis_factory: RedisFactory | None = None) -> None:
        self._url = url
        self._redis_factory: RedisFactory = redis_factory or _default_redis_from_url

    def publish(self, channel: str, message: str) -> int:
        try:
            client = self._redis_factory(self._url)
            out_val: int = int(client.publish(channel, message))
            return out_val
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError) as e:
            logging.getLogger("handwriting_ai").error("redis_publish_error error=%s", str(e))
            raise
