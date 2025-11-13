"""Test Redis Keyspace Notifications as alternative to polling.

This script demonstrates how keyspace notifications can reduce Redis command usage
by being event-driven instead of polling every 2 seconds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime

import redis.asyncio as redis
from redis.exceptions import RedisError


async def test_keyspace_notifications() -> None:
    """Test keyspace notifications with Upstash Redis."""
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        logging.getLogger("handwriting_ai").error("keyspace_demo_missing_url")
        sys.exit(1)

    # Upstash requires TLS - convert redis:// to rediss://
    if redis_url.startswith("redis://"):
        redis_url_tls = redis_url.replace("redis://", "rediss://", 1)
    else:
        redis_url_tls = redis_url

    logging.getLogger("handwriting_ai").info("keyspace_demo_connect url=%s", redis_url_tls[:35])

    # Connect to Redis
    client = await redis.from_url(redis_url_tls, decode_responses=True)

    try:
        # 1. Check current keyspace notification config
        logging.getLogger("handwriting_ai").info("keyspace_demo_step msg=%s", "check_config")
        config = await client.config_get("notify-keyspace-events")
        logging.getLogger("handwriting_ai").info("keyspace_demo_config value=%s", config)

        # 2. Enable keyspace notifications for sorted set operations (RQ uses ZSETs)
        # K = keyspace events, z = sorted set commands, A = all commands
        logging.getLogger("handwriting_ai").info("keyspace_demo_enable msg=%s", "Kz")
        await client.config_set("notify-keyspace-events", "Kz")
        config = await client.config_get("notify-keyspace-events")
        logging.getLogger("handwriting_ai").info("keyspace_demo_config_new value=%s", config)

        # 3. Subscribe to RQ registry changes
        logging.getLogger("handwriting_ai").info("keyspace_demo_step msg=%s", "subscribe")
        pubsub = client.pubsub()

        # Subscribe to all RQ registry keys
        pattern = "__keyspace@0__:rq:registry:*"
        await pubsub.psubscribe(pattern)
        logging.getLogger("handwriting_ai").info("keyspace_demo_subscribed pattern=%s", pattern)

        # 4. Simulate RQ registry changes in another task
        async def simulate_registry_changes() -> None:
            """Simulate RQ adding/removing jobs from registries."""
            await asyncio.sleep(2)

            logging.getLogger("handwriting_ai").info("keyspace_demo_simulate_start")
            test_registry = "rq:registry:failed:digits"

            # Add a job to registry (ZADD)
            logging.getLogger("handwriting_ai").info("keyspace_demo_zadd key=%s", test_registry)
            await client.zadd(test_registry, {"test-job-123": 1234567890.0})
            await asyncio.sleep(1)

            # Remove job from registry (ZREM)
            logging.getLogger("handwriting_ai").info("keyspace_demo_zrem key=%s", test_registry)
            await client.zrem(test_registry, "test-job-123")
            await asyncio.sleep(1)

            # Cleanup
            await client.delete(test_registry)

        # 5. Listen for notifications
        logging.getLogger("handwriting_ai").info("keyspace_demo_listen seconds=%d", 10)

        simulator = asyncio.create_task(simulate_registry_changes())

        timeout = 10
        start = asyncio.get_event_loop().time()
        event_count = 0

        while asyncio.get_event_loop().time() - start < timeout:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "pmessage":
                event_count += 1
                channel = message["channel"]
                data = message["data"]
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                logging.getLogger("handwriting_ai").info(
                    "keyspace_demo_event ts=%s n=%d channel=%s data=%s",
                    timestamp,
                    event_count,
                    channel,
                    data,
                )

        await simulator

        # 6. Results
        logging.getLogger("handwriting_ai").info("keyspace_demo_results n=%d", event_count)

        if event_count >= 2:
            logging.getLogger("handwriting_ai").info("keyspace_demo_success")
        else:
            logging.getLogger("handwriting_ai").warning("keyspace_demo_expected_more_events")

        # Cleanup
        await pubsub.punsubscribe(pattern)
        await pubsub.close()

    except (RedisError, OSError, RuntimeError, ValueError) as e:
        logging.getLogger("handwriting_ai").error("keyspace_demo_error %s", e, exc_info=True)
        import traceback

        traceback.print_exc()
        raise

    finally:
        await client.aclose()


if __name__ == "__main__":
    from handwriting_ai.logging import init_logging

    init_logging()
    asyncio.run(test_keyspace_notifications())
