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
    )
