"""
api/routes/metrics.py

GET /metrics — returns live operational metrics as JSON.

Tracks:
  - total_runs / failed_runs / success_runs
  - avg_latency_ms
  - webhook_attempts / webhook_success / webhook_success_rate
  - queue_backlog
  - job status counts (PENDING / PROCESSING / DONE / FAILED)
"""

from __future__ import annotations

from fastapi import APIRouter, status

from api.dependencies import get_job_tracker, get_metrics, get_task_queue

router = APIRouter(tags=["observability"])


@router.get(
    "/metrics",
    summary="Live pipeline operational metrics",
    status_code=status.HTTP_200_OK,
)
async def get_metrics_endpoint():
    """
    Returns a JSON snapshot of all in-memory operational metrics.
    No authentication required for Alpha (internal use only).
    """
    metrics = get_metrics()
    task_queue = get_task_queue()
    job_tracker = get_job_tracker()

    # Keep queue_backlog fresh at read time
    metrics.set_queue_backlog(task_queue.qsize())

    snapshot = metrics.snapshot()
    snapshot["jobs"] = job_tracker.counts()

    return snapshot
