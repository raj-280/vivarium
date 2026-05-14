"""
api/routes/results.py

GET /results — query historical pipeline results from the database.

Query params:
    cage_id   : filter by cage (optional)
    limit     : max rows returned (default 20, max 100)
    offset    : pagination offset (default 0)
    from_ts   : ISO-8601 start timestamp (optional)
    to_ts     : ISO-8601 end timestamp (optional)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger

from api.schemas.results import ResultsResponse
from core.database import get_session, is_db_ready
from core.repositories import query_pipeline_results

router = APIRouter(tags=["results"])


@router.get(
    "/results",
    summary="Query historical pipeline results",
    response_model=ResultsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_results(
    cage_id: Optional[str] = Query(None, description="Filter by cage ID"),
    limit: int = Query(20, ge=1, le=100, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    from_ts: Optional[datetime] = Query(None, description="Start timestamp (ISO-8601)"),
    to_ts: Optional[datetime] = Query(None, description="End timestamp (ISO-8601)"),
):
    """
    Return paginated pipeline results from the database.

    All query params are optional — omit them to get the latest 20 results.
    """
    if not is_db_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not enabled or not ready",
        )

    logger.info(
        f"GET /results | cage={cage_id} limit={limit} offset={offset} "
        f"from={from_ts} to={to_ts}"
    )

    async with get_session() as session:
        rows = await query_pipeline_results(
            session=session,
            cage_id=cage_id,
            limit=limit,
            offset=offset,
            from_ts=from_ts,
            to_ts=to_ts,
        )

    return ResultsResponse(
        count=len(rows),
        limit=limit,
        offset=offset,
        results=rows,
    )
