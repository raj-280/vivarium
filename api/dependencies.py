"""
api/dependencies.py

Shared FastAPI dependencies — orchestrator, queue, tracker, metrics access.
"""

from __future__ import annotations

from fastapi import Request
from loguru import logger

from core.job_tracker import JobTracker
from core.metrics import MetricsCollector
from core.task_queue import TaskQueue
from pipeline.orchestrator import PipelineOrchestrator


def get_orchestrator(request: Request) -> PipelineOrchestrator:
    """Extract the pipeline orchestrator from app state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        logger.error("PipelineOrchestrator is not initialised — was startup() called?")
        raise RuntimeError("PipelineOrchestrator is not initialised")
    return orchestrator


def get_task_queue() -> TaskQueue:
    """Return the global TaskQueue singleton."""
    from api.main import _task_queue
    if _task_queue is None:
        raise RuntimeError("TaskQueue is not initialised")
    return _task_queue


def get_job_tracker() -> JobTracker:
    """Return the global JobTracker singleton."""
    from api.main import _job_tracker
    if _job_tracker is None:
        raise RuntimeError("JobTracker is not initialised")
    return _job_tracker


def get_metrics() -> MetricsCollector:
    """Return the global MetricsCollector singleton."""
    from api.main import _metrics
    if _metrics is None:
        raise RuntimeError("MetricsCollector is not initialised")
    return _metrics
