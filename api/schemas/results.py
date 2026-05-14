"""
api/schemas/results.py

Pydantic response model for GET /results.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class ResultsResponse(BaseModel):
    count: int
    limit: int
    offset: int
    results: List[Dict[str, Any]]

    model_config = {"json_schema_extra": {
        "example": {
            "count": 2,
            "limit": 20,
            "offset": 0,
            "results": [
                {
                    "result_id": "a1b2c3",
                    "timestamp": "2026-05-11T10:00:00Z",
                    "water_pct": 72.5,
                    "food_pct": 45.0,
                    "mouse_present": True,
                }
            ],
        }
    }}
