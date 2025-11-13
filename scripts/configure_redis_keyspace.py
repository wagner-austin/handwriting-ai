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

from handwriting_ai.logging import get_logger, init_logging


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

    log = get_logger()
    log.info("redis_config_connect url=%s", redis_url[:35])

    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)

        # Test connection
        if not client.ping():
            raise RuntimeError("Redis ping failed")
        log.info("redis_config_ping ok=true")

        # Get current config
        current_config = client.config_get("notify-keyspace-events")
        current_value = current_config.get("notify-keyspace-events", "")
        log.info("redis_config_current value=%s", current_value)

        # Check if keyspace notifications already enabled (order doesn't matter)
        if "K" in current_value and "z" in current_value:
            log.info("redis_config_already_configured value=%s", current_value)
        else:
            # Set keyspace notifications
            target_value = "Kz"
            client.config_set("notify-keyspace-events", target_value)
            log.info("redis_config_set value=%s", target_value)

            # Verify it was set (check contains K and z, not exact match)
            verify_config = client.config_get("notify-keyspace-events")
            verify_value = verify_config.get("notify-keyspace-events", "")
            if "K" not in verify_value or "z" not in verify_value:
                raise RuntimeError(
                    f"Config verification failed: expected K and z, got '{verify_value}'"
                )
            log.info("redis_config_verify value=%s", verify_value)

        # Show what this enables
        log.info("redis_config_enabled_notes details=K:keyspace_events,z:sorted_set_commands")

    except redis.RedisError as e:
        raise RuntimeError(f"Redis configuration failed: {e}") from e
    finally:
        try:
            client.close()
        except (OSError, RuntimeError, ValueError) as e:  # pragma: no cover - defensive
            import logging as _logging

            _logging.getLogger("handwriting_ai").debug("redis_config_close_ignored error=%s", e)


def main() -> None:
    """Run keyspace notification configuration from environment."""
    init_logging()
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        get_logger().error("redis_config_missing_url")
        sys.exit(1)

    try:
        configure_keyspace_notifications(redis_url)
        # Human-friendly confirmation via structured logging
        get_logger().info("Configuration complete")
    except (RuntimeError, OSError, ValueError) as e:
        # Log via structured logger and re-raise to ensure non-zero exit without silent except
        get_logger().error("redis_keyspace_config_failed error=%s", e)
        raise


if __name__ == "__main__":  # pragma: no cover
    # Delegate to the importable module to honor monkeypatching in tests
    # (runpy.run_module executes this file as __main__, separate from
    # scripts.configure_redis_keyspace already imported and patched).
    import importlib as _importlib

    _mod = _importlib.import_module("scripts.configure_redis_keyspace")
    _mod.main()
