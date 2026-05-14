"""
core/repositories.py

Database write and read operations for pipeline_results and alert_log.

YOLOX update: insert_pipeline_result now persists bedding fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import AlertLogRow, PipelineResultRow
from core.result import AlertRecord, PipelineResult


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


async def insert_pipeline_result(
    session: AsyncSession,
    result: PipelineResult,
    cage_id: str,
) -> Optional[str]:
    """Write one pipeline run to pipeline_results. Returns the UUID or None on failure."""
    row_id = _new_id()
    try:
        row = PipelineResultRow(
            id=row_id,
            cage_id=cage_id,
            image_path=result.image_path,
            water_pct=result.water_pct,
            food_pct=result.food_pct,
            mouse_present=result.mouse_present,
            water_confidence=result.water_confidence,
            food_confidence=result.food_confidence,
            mouse_confidence=result.mouse_confidence,
            bedding_condition=result.bedding_condition,
            bedding_confidence=result.bedding_confidence,
            bedding_area_pct=result.bedding_area_pct,
            uncertain_targets=result.uncertain_targets,
            raw_detections=result.raw_detections,
            processed_at=_now(),
        )
        session.add(row)
        await session.commit()
        logger.debug(f"pipeline_results insert OK | id={row_id} | cage={cage_id}")
        return row_id
    except Exception as exc:
        await session.rollback()
        logger.error(f"pipeline_results insert failed | cage={cage_id} | {exc}")
        return None


async def query_pipeline_results(
    session: AsyncSession,
    cage_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    from_ts: Optional[datetime] = None,
    to_ts: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Query pipeline_results with optional filters."""
    try:
        stmt = select(PipelineResultRow).order_by(
            PipelineResultRow.processed_at.desc()
        )
        if cage_id:
            stmt = stmt.where(PipelineResultRow.cage_id == cage_id)
        if from_ts:
            stmt = stmt.where(PipelineResultRow.processed_at >= from_ts)
        if to_ts:
            stmt = stmt.where(PipelineResultRow.processed_at <= to_ts)

        stmt = stmt.limit(limit).offset(offset)
        rows = await session.execute(stmt)
        results = rows.scalars().all()

        return [
            {
                "result_id":          row.id,
                "cage_id":            row.cage_id,
                "image_path":         row.image_path,
                "water_pct":          row.water_pct,
                "food_pct":           row.food_pct,
                "mouse_present":      row.mouse_present,
                "water_confidence":   row.water_confidence,
                "food_confidence":    row.food_confidence,
                "mouse_confidence":   row.mouse_confidence,
                "bedding_condition":  row.bedding_condition,
                "bedding_confidence": row.bedding_confidence,
                "bedding_area_pct":   row.bedding_area_pct,
                "uncertain_targets":  row.uncertain_targets,
                "raw_detections":     row.raw_detections,
                "timestamp":          row.processed_at.isoformat() if row.processed_at else None,
            }
            for row in results
        ]
    except Exception as exc:
        logger.error(f"pipeline_results query failed | {exc}")
        return []


async def insert_alert(
    session: AsyncSession,
    alert: AlertRecord,
    cage_id: str,
) -> Optional[str]:
    """Write a fired alert to alert_log. Returns the UUID or None on failure."""
    row_id = _new_id()
    try:
        row = AlertLogRow(
            id=row_id,
            cage_id=cage_id,
            target=alert.target,
            alert_type=alert.alert_type,
            value=alert.value,
            message=alert.message,
            fired_at=_now(),
        )
        session.add(row)
        await session.commit()
        logger.debug(
            f"alert_log insert OK | id={row_id} "
            f"| target={alert.target} type={alert.alert_type} | cage={cage_id}"
        )
        return row_id
    except Exception as exc:
        await session.rollback()
        logger.error(f"alert_log insert failed | target={alert.target} | {exc}")
        return None
