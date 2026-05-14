"""
core/job_tracker.py

In-memory job status tracker.

Tracks the lifecycle of every async pipeline job:
    PENDING    → job accepted, waiting in queue
    PROCESSING → worker picked it up, pipeline running
    DONE       → pipeline finished successfully
    FAILED     → pipeline raised an exception

Thread/coroutine safe: uses a plain dict — fine for asyncio single-thread model.

Usage:
    tracker = JobTracker()
    tracker.set_pending(request_id)
    tracker.set_processing(request_id)
    tracker.set_done(request_id, result)

    entry = tracker.get(request_id)   # → JobEntry | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from loguru import logger

from core.result import PipelineResult


class JobStatus:
    PENDING    = "PENDING"
    PROCESSING = "PROCESSING"
    DONE       = "DONE"
    FAILED     = "FAILED"


@dataclass
class JobEntry:
    request_id: str
    cage_id: str
    status: str                          # JobStatus value
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    result: Optional[PipelineResult] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        base = {
            "request_id": self.request_id,
            "cage_id":    self.cage_id,
            "status":     self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error":      self.error,
        }
        if self.result is not None:
            base["result"] = self.result.to_dict(cage_id=self.cage_id)
        return base


class JobTracker:
    """
    In-memory store for job status entries.

    For Alpha: plain dict, lives in process memory.
    For production: swap backing store to Redis or a DB table.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobEntry] = {}

    def set_pending(self, request_id: str, cage_id: str) -> JobEntry:
        entry = JobEntry(
            request_id=request_id,
            cage_id=cage_id,
            status=JobStatus.PENDING,
        )
        self._jobs[request_id] = entry
        logger.debug(f"Job PENDING | request_id={request_id} | cage={cage_id}")
        return entry

    def set_processing(self, request_id: str) -> None:
        entry = self._jobs.get(request_id)
        if entry:
            entry.status = JobStatus.PROCESSING
            entry.updated_at = datetime.now(tz=timezone.utc)
            logger.debug(f"Job PROCESSING | request_id={request_id}")

    def set_done(self, request_id: str, result: PipelineResult) -> None:
        entry = self._jobs.get(request_id)
        if entry:
            entry.status = JobStatus.DONE
            entry.result = result
            entry.updated_at = datetime.now(tz=timezone.utc)
            logger.debug(f"Job DONE | request_id={request_id}")

    def set_failed(self, request_id: str, error: str) -> None:
        entry = self._jobs.get(request_id)
        if entry:
            entry.status = JobStatus.FAILED
            entry.error = error
            entry.updated_at = datetime.now(tz=timezone.utc)
            logger.warning(f"Job FAILED | request_id={request_id} | error={error}")

    def get(self, request_id: str) -> Optional[JobEntry]:
        return self._jobs.get(request_id)

    def counts(self) -> Dict[str, int]:
        """Return count per status — used by metrics endpoint."""
        result = {
            JobStatus.PENDING:    0,
            JobStatus.PROCESSING: 0,
            JobStatus.DONE:       0,
            JobStatus.FAILED:     0,
        }
        for entry in self._jobs.values():
            result[entry.status] = result.get(entry.status, 0) + 1
        return result
