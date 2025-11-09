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
        print("ERROR: REDIS_URL environment variable not set")
        sys.exit(1)

    # Upstash requires TLS - convert redis:// to rediss://
    if redis_url.startswith("redis://"):
        redis_url_tls = redis_url.replace("redis://", "rediss://", 1)
    else:
        redis_url_tls = redis_url

    print(f"Connecting to Redis: {redis_url_tls[:35]}...")

    # Connect to Redis
    client = await redis.from_url(redis_url_tls, decode_responses=True)

    try:
        # 1. Check current keyspace notification config
        print("\n1. Checking current keyspace notification configuration...")
        config = await client.config_get("notify-keyspace-events")
        print(f"   Current config: {config}")

        # 2. Enable keyspace notifications for sorted set operations (RQ uses ZSETs)
        # K = keyspace events, z = sorted set commands, A = all commands
        print("\n2. Enabling keyspace notifications (Kz = keyspace + sorted sets)...")
        await client.config_set("notify-keyspace-events", "Kz")
        config = await client.config_get("notify-keyspace-events")
        print(f"   New config: {config}")

        # 3. Subscribe to RQ registry changes
        print("\n3. Setting up subscription to RQ registry changes...")
        pubsub = client.pubsub()

        # Subscribe to all RQ registry keys
        pattern = "__keyspace@0__:rq:registry:*"
        await pubsub.psubscribe(pattern)
        print(f"   Subscribed to pattern: {pattern}")

        # 4. Simulate RQ registry changes in another task
        async def simulate_registry_changes() -> None:
            """Simulate RQ adding/removing jobs from registries."""
            await asyncio.sleep(2)

            print("\n4. Simulating registry changes...")
            test_registry = "rq:registry:failed:digits"

            # Add a job to registry (ZADD)
            print(f"   Adding job to {test_registry}...")
            await client.zadd(test_registry, {"test-job-123": 1234567890.0})
            await asyncio.sleep(1)

            # Remove job from registry (ZREM)
            print(f"   Removing job from {test_registry}...")
            await client.zrem(test_registry, "test-job-123")
            await asyncio.sleep(1)

            # Cleanup
            await client.delete(test_registry)

        # 5. Listen for notifications
        print("\n5. Listening for notifications (10 seconds)...")
        print("   (Events will appear as they occur)")

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
                print(f"   [{timestamp}] Event #{event_count}: {channel} -> {data}")

        await simulator

        # 6. Results
        print("\n6. Results:")
        print(f"   Total events received: {event_count}")
        print("   Events expected: 2 (zadd + zrem)")

        if event_count >= 2:
            print("\n[SUCCESS] Keyspace notifications working!")
            print("\nBenefits:")
            print("  - Zero Redis commands when no changes occur")
            print("  - Instant notification when registries change")
            print("  - Current polling: ~130 commands/minute idle")
            print("  - With notifications: ~0 commands/minute idle")
        else:
            print("\n[WARNING] Expected more events")
            print("   This might indicate notifications aren't fully enabled")

        # Cleanup
        await pubsub.punsubscribe(pattern)
        await pubsub.close()

    except (RedisError, OSError, RuntimeError, ValueError) as e:
        logging.getLogger("handwriting_ai").error("keyspace_demo_error %s", e, exc_info=True)
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        await client.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(test_keyspace_notifications())
