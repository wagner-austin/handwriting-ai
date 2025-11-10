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

    class RedisPubSubProto(Protocol):  # pragma: no cover - typing only
        def psubscribe(self, *patterns: str) -> None: ...
        def get_message(
            self, ignore_subscribe_messages: bool = True, timeout: float | None = None
        ) -> dict[str, object] | None: ...
        def close(self) -> None: ...

    def redis_pubsub(url: str) -> RedisPubSubProto: ...
    def redis_config_get(url: str, param: str) -> dict[str, str]: ...
    def redis_db_index(url: str) -> int: ...

else:  # pragma: no cover - runtime only
    import logging

    def redis_from_url(url: str, *, decode_responses: bool = False) -> RedisClient:
        import redis

        return redis.Redis.from_url(url, decode_responses=decode_responses)

    def rq_connect(url: str) -> RedisDebugClientProto:
        import redis

        # RQ stores binary (pickled) payloads; decode_responses must be False
        return redis.Redis.from_url(
            url,
            decode_responses=False,
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

    # --- Redis pubsub + config helpers for notification watcher ---
    def redis_pubsub(url: str) -> RedisPubSubProto:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        return client.pubsub()

    def redis_config_get(url: str, param: str) -> dict[str, str]:
        import redis

        client = redis.Redis.from_url(url, decode_responses=True)
        try:
            out = client.config_get(param)
            if not isinstance(out, dict):
                return {}
            # Cast keys/values to str
            return {str(k): str(v) for k, v in out.items()}
        finally:
            from contextlib import suppress

            with suppress(Exception):  # pragma: no cover - best effort
                client.close()

    def redis_db_index(url: str) -> int:
        # Parse db index from URL; default 0
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.path and parsed.path.strip("/").isdigit():
            return int(parsed.path.strip("/"))
        return 0

    # no additional helpers required
