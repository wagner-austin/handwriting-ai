#!/usr/bin/env python3
"""Configure Redis keyspace notifications for job watcher.

Usage:
    python scripts/configure_redis_keyspace.py
    # Or with explicit URL:
    REDIS_URL=redis://... python scripts/configure_redis_keyspace.py
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress

from handwriting_ai.logging import get_logger


def configure_keyspace_notifications(redis_url: str) -> None:
    """Configure Redis to publish keyspace notifications for sorted set events.

    Sets notify-keyspace-events=Kz:
    - K: Keyspace events (publishes __keyspace@<db>__:<key> notifications)
    - z: Sorted set commands (RQ uses sorted sets for job registries)

    Args:
        redis_url: Redis connection URL

    Raises:
        RuntimeError: If configuration fails
    """
    import redis

    print(f"Connecting to Redis: {redis_url[:35]}...")

    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)

        # Test connection
        if not client.ping():
            raise RuntimeError("Redis ping failed")
        print("Connected to Redis")

        # Get current config
        current_config = client.config_get("notify-keyspace-events")
        current_value = current_config.get("notify-keyspace-events", "")
        print(f"Current notify-keyspace-events: '{current_value}'")

        # Check if keyspace notifications already enabled (order doesn't matter)
        if "K" in current_value and "z" in current_value:
            print(f"Already configured with '{current_value}' (contains both K and z)")
        else:
            # Set keyspace notifications
            target_value = "Kz"
            client.config_set("notify-keyspace-events", target_value)
            print(f"Set notify-keyspace-events to '{target_value}'")

            # Verify it was set (check contains K and z, not exact match)
            verify_config = client.config_get("notify-keyspace-events")
            verify_value = verify_config.get("notify-keyspace-events", "")
            if "K" not in verify_value or "z" not in verify_value:
                raise RuntimeError(
                    f"Config verification failed: expected K and z, got '{verify_value}'"
                )
            print(f"Verified configuration: '{verify_value}'")

        # Show what this enables
        print("\nKeyspace notifications enabled:")
        print("  K = Keyspace events (publishes on key changes)")
        print("  z = Sorted set commands (RQ job registries)")
        print("\nWatcher can now subscribe to: __keyspace@0__:rq:registry:*:queue")

    except redis.RedisError as e:
        raise RuntimeError(f"Redis configuration failed: {e}") from e
    finally:
        with suppress(Exception):
            client.close()


def main() -> None:
    """Run keyspace notification configuration from environment."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        print("ERROR: REDIS_URL environment variable not set", file=sys.stderr)
        print("\nUsage:", file=sys.stderr)
        print("  REDIS_URL=redis://... python scripts/configure_redis_keyspace.py", file=sys.stderr)
        sys.exit(1)

    try:
        configure_keyspace_notifications(redis_url)
        print("\nConfiguration complete")
    except (RuntimeError, OSError, ValueError) as e:
        # Log via structured logger and re-raise to ensure non-zero exit without silent except
        get_logger().error("redis_keyspace_config_failed error=%s", e)
        raise


if __name__ == "__main__":
    main()
