#!/usr/bin/env python3
"""Quick Redis inspection script."""

from __future__ import annotations

import logging
import os
import sys

import redis


def main() -> None:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        print("ERROR: REDIS_URL not set")
        sys.exit(1)

    print(f"Connecting to Redis: {url[:35]}...")

    r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)

    try:
        # Test connection
        print(f"Ping: {r.ping()}")

        # Check keyspace notifications
        cfg = r.config_get("notify-keyspace-events")
        print(f"notify-keyspace-events: '{cfg.get('notify-keyspace-events', '')}'")

        # Database stats
        print(f"\nTotal keys: {r.dbsize()}")

        # RQ queues
        queues = r.keys("rq:queue:*")
        print(f"\nRQ Queues ({len(queues)}):")
        for q in queues[:10]:
            print(f"  {q}")

        # RQ registries
        registries = r.keys("rq:registry:*")
        print(f"\nRQ Registries ({len(registries)}):")
        for reg in sorted(registries)[:15]:
            count = r.zcard(reg) if r.type(reg) == "zset" else 0
            print(f"  {reg} ({count} jobs)")

        # Sample jobs
        jobs = r.keys("rq:job:*")
        print(f"\nRQ Jobs: {len(jobs)} total")

        # Show all keys
        all_keys = r.keys("*")
        print(f"\nAll keys ({len(all_keys)}):")
        for key in all_keys[:20]:
            key_type = r.type(key)
            print(f"  {key} ({key_type})")

    except (OSError, ConnectionError, redis.exceptions.RedisError, RuntimeError, ValueError) as e:
        logging.getLogger("handwriting_ai").error("redis_check_error error=%s", e)
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
