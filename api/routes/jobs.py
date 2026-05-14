"""
api/routes/jobs.py

GET /job/{request_id} — returns the status and result of an async pipeline job.

Status values:
    PENDING    → queued, not yet started
    PROCESSING → pipeline is running
    DONE       → finished, result included
    FAILED     → pipeline error, error message included
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.dependencies import get_job_tracker

router = APIRouter(tags=["jobs"])


@router.get(
    "/job/{request_id}",
    summary="Get async job status and result",
    status_code=status.HTTP_200_OK,
)
async def get_job(request_id: str):
    """
    Poll the status of a job submitted via POST /analyze.

    Returns status + full result when DONE, or error message when FAILED.
    """
    tracker = get_job_tracker()
    entry = tracker.get(request_id)

    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{request_id}' not found",
        )

    return entry.to_dict()
