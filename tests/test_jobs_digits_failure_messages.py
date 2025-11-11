from __future__ import annotations

# Tests removed: process_train_job() no longer publishes failed events
# Failure notifications are now published exclusively by the watcher
# when it detects jobs in the failed/canceled registries via keyspace events
