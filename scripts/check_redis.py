#!/usr/bin/env python3
"""Quick Redis inspection script using structured logging."""

from __future__ import annotations

import logging
import os
import sys

import redis

from handwriting_ai.logging import get_logger, init_logging


def main() -> None:
    # Initialize structured logging for CLI output
    init_logging()
    log = get_logger()
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        log.error("redis_check_missing_url")
        sys.exit(1)

    log.info("redis_check_connect url=%s", url[:35])

    r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)

    try:
        # Test connection
        log.info("redis_check_ping_ok ok=%s", r.ping())

        # Check keyspace notifications
        cfg = r.config_get("notify-keyspace-events")
        log.info("redis_check_notify_keyspace value=%s", cfg.get("notify-keyspace-events", ""))

        # Database stats
        log.info("redis_check_total_keys n=%d", r.dbsize())

        # RQ queues
        queues = r.keys("rq:queue:*")
        log.info("redis_check_rq_queues n=%d", len(queues))
        for q in queues[:10]:
            log.info("redis_check_rq_queue name=%s", q)

        # RQ registries
        registries = r.keys("rq:registry:*")
        log.info("redis_check_rq_registries n=%d", len(registries))
        for reg in sorted(registries)[:15]:
            count = r.zcard(reg) if r.type(reg) == "zset" else 0
            log.info("redis_check_rq_registry name=%s jobs=%d", reg, count)

        # Sample jobs
        jobs = r.keys("rq:job:*")
        log.info("redis_check_rq_jobs n=%d", len(jobs))

        # Show all keys
        all_keys = r.keys("*")
        log.info("redis_check_all_keys n=%d", len(all_keys))
        for key in all_keys[:20]:
            key_type = r.type(key)
            log.info("redis_check_key name=%s type=%s", key, key_type)
        # Ensure logs are flushed before exiting so tests can capture output
        from contextlib import suppress as _suppress

        for _h in list(logging.getLogger("handwriting_ai").handlers):
            if hasattr(_h, "flush"):
                with _suppress(Exception):
                    _h.flush()

    except (OSError, ConnectionError, redis.exceptions.RedisError, RuntimeError, ValueError) as e:
        logging.getLogger("handwriting_ai").error("redis_check_error error=%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
