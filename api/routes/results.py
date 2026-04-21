"""
api/routes/results.py

GET /results — query historical pipeline results from the database.

FIX 1: Removed duplicated _authenticate / _get_orchestrator.
        Both now imported from api.dependencies.

FIX 2: Route no longer accesses orchestrator._storage directly (private attribute).
        Now calls orchestrator.get_results() — a public method added to the orchestrator.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from api.dependencies import authenticate, get_orchestrator
from core.config_loader import get_config
from pipeline.orchestrator import PipelineOrchestrator

router = APIRouter(tags=["results"])


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
    _: str = Depends(authenticate),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
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
        # FIX: use the public orchestrator method — never access _storage directly
        rows: List[dict] = await orchestrator.get_results(
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