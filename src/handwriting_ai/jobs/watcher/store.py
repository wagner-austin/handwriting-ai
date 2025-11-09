from __future__ import annotations

import logging
from typing import Protocol

from .adapters import redis_from_url as _default_redis_from_url
from .ports import RedisFactory


class ProcessedStore(Protocol):
    def seen(self, job_id: str) -> bool: ...
    def mark(self, job_id: str) -> None: ...


class RedisProcessedStore:
    def __init__(self, url: str, *, key: str, redis_factory: RedisFactory | None = None) -> None:
        self._url = url
        self._key = key
        self._redis_factory: RedisFactory = redis_factory or _default_redis_from_url

    def seen(self, job_id: str) -> bool:
        try:
            client = self._redis_factory(self._url)
            val: bool = bool(client.sismember(self._key, job_id))
            return val
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError) as e:
            logging.getLogger("handwriting_ai").error("redis_seen_error error=%s", str(e))
            raise

    def mark(self, job_id: str) -> None:
        try:
            client = self._redis_factory(self._url)
            client.sadd(self._key, job_id)
        except (OSError, RuntimeError, ValueError, TypeError, ConnectionError) as e:
            logging.getLogger("handwriting_ai").error("redis_mark_error error=%s", str(e))
            raise
