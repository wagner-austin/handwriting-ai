from __future__ import annotations

# Tests removed: custom exception handler no longer exists
# RQ's default handler now adds failed jobs to FailedJobRegistry
# Watcher detects these via keyspace events and publishes notifications
