from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:

    class RedisClient(Protocol):  # pragma: no cover - typing only
        def publish(self, channel: str, message: str) -> int: ...
        def sismember(self, key: str, member: str) -> bool: ...
        def sadd(self, key: str, *members: str) -> int: ...

    def redis_from_url(url: str, *, decode_responses: bool = False) -> RedisClient: ...

    class RQQueueProto(Protocol):  # pragma: no cover - typing only
        ...

    class RQRegistryProto(Protocol):  # pragma: no cover - typing only
        def get_job_ids(self) -> list[str]: ...

    class RQJobProto(Protocol):  # pragma: no cover - typing only
        args: object
        exc_info: object
        started_at: object
        enqueued_at: object

    class RedisDebugClientProto(Protocol):  # pragma: no cover - typing only
        def zrange(self, key: str, start: int, end: int) -> list[str] | list[bytes]: ...
        def hget(self, key: str, field: str) -> str | bytes | None: ...

    def rq_connect(url: str) -> RedisDebugClientProto: ...
    def rq_queue(conn: RedisDebugClientProto, name: str) -> RQQueueProto: ...
    def rq_failed_registry(queue: RQQueueProto) -> RQRegistryProto: ...
    def rq_started_registry(queue: RQQueueProto) -> RQRegistryProto: ...
    def rq_canceled_registry(queue: RQQueueProto) -> RQRegistryProto: ...
    def rq_fetch_job(conn: RedisDebugClientProto, job_id: str) -> RQJobProto: ...

else:  # pragma: no cover - runtime only
    import logging

    def redis_from_url(url: str, *, decode_responses: bool = False) -> RedisClient:
        import redis

        return redis.Redis.from_url(url, decode_responses=decode_responses)

    def rq_connect(url: str) -> RedisDebugClientProto:
        import redis

        return redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=10.0,
        )

    def rq_queue(conn: RedisDebugClientProto, name: str) -> RQQueueProto:
        import rq

        return rq.Queue(name, connection=conn)

    def rq_failed_registry(queue: RQQueueProto) -> RQRegistryProto:
        import rq

        return rq.registry.FailedJobRegistry(queue=queue)

    def rq_started_registry(queue: RQQueueProto) -> RQRegistryProto:
        import rq

        return rq.registry.StartedJobRegistry(queue=queue)

    def rq_canceled_registry(queue: RQQueueProto) -> RQRegistryProto:
        import rq

        if not hasattr(queue, "name"):
            logging.getLogger("handwriting_ai").error("rq_canceled_registry_invalid_queue")
            raise TypeError("Invalid RQ queue provided")
        reg = getattr(rq, "registry", None)
        cls = getattr(reg, "CanceledJobRegistry", None)
        if cls is None:
            logging.getLogger("handwriting_ai").error("rq_canceled_registry_unavailable")
            raise RuntimeError("RQ CanceledJobRegistry is unavailable")
        return cls(queue=queue)

    def rq_fetch_job(conn: RedisDebugClientProto, job_id: str) -> RQJobProto:
        import rq

        return rq.job.Job.fetch(job_id, connection=conn)
