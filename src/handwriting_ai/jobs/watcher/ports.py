from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from logging import Logger
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # typing-only forward references without import-time deps
    from .adapters import (
        RedisClient as _RedisClient,
    )
    from .adapters import (
        RedisDebugClientProto as _RedisDebugClientProto,
    )
    from .adapters import (
        RedisPubSubProto as _RedisPubSubProto,
    )
    from .adapters import (
        RQJobProto as _RQJobProto,
    )
    from .adapters import (
        RQQueueProto as _RQQueueProto,
    )
    from .adapters import (
        RQRegistryProto as _RQRegistryProto,
    )


class RedisFactory(Protocol):  # pragma: no cover - typing only
    def __call__(self, url: str, *, decode_responses: bool = False) -> _RedisClient: ...


# Defaults for notification helpers to keep existing tests simple
def _noop_pubsub(_: str) -> _RedisPubSubProto:  # pragma: no cover - tests that don't use notify
    class _P:
        def psubscribe(self, *patterns: str) -> None:
            return None

        def get_message(
            self,
            ignore_subscribe_messages: bool = True,
            timeout: float | None = None,
        ) -> dict[str, object] | None:
            return None

        def close(self) -> None:
            return None

    return _P()


def _noop_cfg(_: str, __: str) -> dict[str, str]:  # pragma: no cover
    return {}


def _noop_db(_: str) -> int:  # pragma: no cover
    return 0


@dataclass(frozen=True)
class WatcherPorts:
    # Redis + RQ integration points
    redis_from_url: RedisFactory
    rq_connect: Callable[[str], _RedisDebugClientProto]
    rq_queue: Callable[[_RedisDebugClientProto, str], _RQQueueProto]
    rq_failed_registry: Callable[[_RQQueueProto], _RQRegistryProto]
    rq_started_registry: Callable[[_RQQueueProto], _RQRegistryProto]
    rq_canceled_registry: Callable[[_RQQueueProto], _RQRegistryProto]
    rq_fetch_job: Callable[[_RedisDebugClientProto, str], _RQJobProto]

    # Logic helpers
    coerce_job_ids: Callable[[Sequence[object]], list[str]]
    extract_payload: Callable[[object], dict[str, object]]
    detect_failed_reason: Callable[[object, str], str | None]
    summarize_exc_info: Callable[[object], str]
    make_logger: Callable[[], Logger]
    # Redis keyspace notification helpers (defaults for tests that don't use them)
    redis_pubsub: Callable[[str], _RedisPubSubProto] = _noop_pubsub
    redis_config_get: Callable[[str, str], dict[str, str]] = _noop_cfg
    redis_db_index: Callable[[str], int] = _noop_db


def make_default_ports() -> WatcherPorts:
    from . import adapters, logic

    return WatcherPorts(
        redis_from_url=adapters.redis_from_url,
        rq_connect=adapters.rq_connect,
        rq_queue=adapters.rq_queue,
        rq_failed_registry=adapters.rq_failed_registry,
        rq_started_registry=adapters.rq_started_registry,
        rq_canceled_registry=adapters.rq_canceled_registry,
        rq_fetch_job=adapters.rq_fetch_job,
        coerce_job_ids=logic.coerce_job_ids,
        extract_payload=logic.extract_payload,
        detect_failed_reason=logic.detect_failed_reason,
        summarize_exc_info=logic.summarize_exc_info,
        make_logger=logic.make_logger,
        redis_pubsub=adapters.redis_pubsub,
        redis_config_get=adapters.redis_config_get,
        redis_db_index=adapters.redis_db_index,
    )
