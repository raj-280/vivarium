"""
core/task_queue.py

Async task queue for non-blocking image processing.

Uses Python asyncio.Queue — no Celery/Redis needed for Alpha.
N worker coroutines pull jobs from the queue and run the pipeline.

Config block in config.yaml:

    queue:
      workers: 4       # number of concurrent worker coroutines
      maxsize: 100     # max items in queue before back-pressure (0 = unlimited)

Flow:
    POST /analyze → enqueue(job) → return request_id immediately
    worker loop  → dequeue job   → orchestrator.run() → job_tracker.set_done()
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

from dotmap import DotMap
from loguru import logger

if TYPE_CHECKING:
    from core.job_tracker import JobTracker
    from pipeline.orchestrator import PipelineOrchestrator


@dataclass
class QueueJob:
    """One item in the task queue."""
    request_id: str
    image_bytes: bytes
    filename: str
    cage_id: str


class TaskQueue:
    """
    Async task queue with a configurable worker pool.

    Instantiate once at startup, call startup() to launch workers,
    shutdown() to drain and stop them.
    """

    def __init__(self, config: DotMap) -> None:
        queue_cfg = getattr(config, "queue", DotMap())
        self._num_workers: int = int(getattr(queue_cfg, "workers", 4))
        maxsize: int = int(getattr(queue_cfg, "maxsize", 100))

        self._queue: asyncio.Queue[QueueJob] = asyncio.Queue(maxsize=maxsize)
        self._worker_tasks: List[asyncio.Task] = []

        logger.info(
            f"TaskQueue initialised | workers={self._num_workers} | maxsize={maxsize}"
        )

    def qsize(self) -> int:
        """Current number of items waiting in the queue."""
        return self._queue.qsize()

    async def enqueue(
        self,
        image_bytes: bytes,
        filename: str,
        cage_id: str,
    ) -> str:
        """
        Add a job to the queue and return its request_id immediately.
        Raises asyncio.QueueFull if the queue is at capacity.
        """
        request_id = str(uuid.uuid4())
        job = QueueJob(
            request_id=request_id,
            image_bytes=image_bytes,
            filename=filename,
            cage_id=cage_id,
        )
        self._queue.put_nowait(job)
        logger.info(
            f"Job enqueued | request_id={request_id} | cage={cage_id} "
            f"| queue_size={self._queue.qsize()}"
        )
        return request_id

    async def startup(
        self,
        orchestrator: "PipelineOrchestrator",
        job_tracker: "JobTracker",
    ) -> None:
        """Launch N worker coroutines."""
        for i in range(self._num_workers):
            task = asyncio.create_task(
                self._worker(i, orchestrator, job_tracker),
                name=f"queue-worker-{i}",
            )
            self._worker_tasks.append(task)
        logger.info(f"TaskQueue started | {self._num_workers} workers running")

    async def shutdown(self) -> None:
        """
        Send sentinel None values to stop workers, then cancel and await them.
        """
        logger.info("TaskQueue shutting down — draining workers")
        for _ in self._worker_tasks:
            await self._queue.put(None)  # type: ignore[arg-type]

        for task in self._worker_tasks:
            try:
                await asyncio.wait_for(task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        self._worker_tasks.clear()
        logger.info("TaskQueue shutdown complete")

    async def _worker(
        self,
        worker_id: int,
        orchestrator: "PipelineOrchestrator",
        job_tracker: "JobTracker",
    ) -> None:
        """Worker loop — pulls jobs from queue and runs pipeline."""
        logger.info(f"Queue worker-{worker_id} started")
        while True:
            job = await self._queue.get()

            # Sentinel — time to stop
            if job is None:
                self._queue.task_done()
                break

            logger.info(
                f"Worker-{worker_id} picked up job | request_id={job.request_id} "
                f"| cage={job.cage_id}"
            )
            job_tracker.set_processing(job.request_id)

            try:
                result = await orchestrator.run(
                    job.image_bytes, job.filename, job.cage_id
                )
                job_tracker.set_done(job.request_id, result)
                logger.info(
                    f"Worker-{worker_id} completed job | request_id={job.request_id}"
                )
            except Exception as exc:
                logger.error(
                    f"Worker-{worker_id} job failed | request_id={job.request_id} | {exc}"
                )
                job_tracker.set_failed(job.request_id, str(exc))
            finally:
                self._queue.task_done()

        logger.info(f"Queue worker-{worker_id} stopped")
