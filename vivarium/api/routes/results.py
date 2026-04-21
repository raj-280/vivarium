"""
api/routes/results.py

GET /results — query historical pipeline results from the database.

Authentication: X-API-Key header.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from core.config_loader import get_config
from pipeline.orchestrator import PipelineOrchestrator

router = APIRouter(tags=["results"])


def _get_orchestrator(request: Request) -> PipelineOrchestrator:
    return request.app.state.orchestrator


def _authenticate(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    config = get_config()
    expected_key: str = config.api.api_key
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


@router.get(
    "/results",
    summary="Query historical pipeline results",
    response_description="Ordered list of pipeline results from the database",
    status_code=status.HTTP_200_OK,
)
async def get_results(
    request: Request,
    limit: int = Query(default=20, ge=1, le=200, description="Max results to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    target: Optional[str] = Query(
        default=None,
        description="Filter by target: 'water', 'food', or 'mouse'",
    ),
    _: str = Depends(_authenticate),
    orchestrator: PipelineOrchestrator = Depends(_get_orchestrator),
):
    """
    Return historical pipeline results from the database.

    Results are ordered by **processed_at DESC** (newest first).

    - **limit**: Number of records to return (default 20, max 200)
    - **offset**: Pagination offset
    - **target**: Optional filter — only return rows where this target was measured
    """
    config = get_config()

    allowed_targets = set(config.targets.enabled) | {None}
    if target is not None and target not in allowed_targets:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown target '{target}'. Allowed: {list(config.targets.enabled)}",
        )

    try:
        rows: List[dict] = await orchestrator._storage.get_results(
            limit=limit, offset=offset, target=target
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(exc)}",
        )

    return {
        "count": len(rows),
        "limit": limit,
        "offset": offset,
        "results": rows,
    }
