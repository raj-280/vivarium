"""
core/metrics.py

In-memory metrics collector.

Tracks:
  - total_runs         : total pipeline executions
  - failed_runs        : runs that raised an error or were rejected
  - webhook_success    : successful webhook POSTs
  - webhook_attempts   : total webhook POST attempts
  - total_latency_ms   : sum of all pipeline latencies (for avg calculation)
  - queue_backlog      : live queue depth (set externally)

All counters are integers — simple, no external dependencies.
Exposed via GET /metrics.

Usage:
    metrics = MetricsCollector()
    metrics.record_run(success=True, latency_ms=142.5)
    metrics.record_webhook(success=True)
    metrics.set_queue_backlog(q.qsize())
"""

from __future__ import annotations

import threading
from typing import Dict, Any


class MetricsCollector:
    """
    Thread-safe in-memory metrics collector.
    Uses a lock so concurrent async workers don't race on counters.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_runs: int = 0
        self._failed_runs: int = 0
        self._webhook_success: int = 0
        self._webhook_attempts: int = 0
        self._total_latency_ms: float = 0.0
        self._queue_backlog: int = 0

    def record_run(self, *, success: bool, latency_ms: float) -> None:
        with self._lock:
            self._total_runs += 1
            if not success:
                self._failed_runs += 1
            self._total_latency_ms += latency_ms

    def record_webhook(self, *, success: bool) -> None:
        with self._lock:
            self._webhook_attempts += 1
            if success:
                self._webhook_success += 1

    def set_queue_backlog(self, size: int) -> None:
        with self._lock:
            self._queue_backlog = size

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot of all current metrics."""
        with self._lock:
            avg_latency = (
                round(self._total_latency_ms / self._total_runs, 2)
                if self._total_runs > 0
                else 0.0
            )
            webhook_success_rate = (
                round(self._webhook_success / self._webhook_attempts, 4)
                if self._webhook_attempts > 0
                else None
            )
            return {
                "total_runs":          self._total_runs,
                "failed_runs":         self._failed_runs,
                "success_runs":        self._total_runs - self._failed_runs,
                "avg_latency_ms":      avg_latency,
                "webhook_attempts":    self._webhook_attempts,
                "webhook_success":     self._webhook_success,
                "webhook_success_rate": webhook_success_rate,
                "queue_backlog":       self._queue_backlog,
            }
