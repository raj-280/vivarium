"""
core/alerting.py

Threshold evaluation and alert persistence.

Checks water and food levels after each pipeline run.
If LOW or EMPTY → writes a row to alert_log.

Config block in config.yaml:

    alerts:
      enabled: true
      thresholds:
        water:
          low: 20.0
          empty: 5.0
        food:
          low: 20.0
          empty: 5.0
"""

from __future__ import annotations

from typing import List, Optional

from dotmap import DotMap
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from core.repositories import insert_alert
from core.result import AlertRecord, PipelineResult

_UNDETERMINED = "UNDETERMINED"


class AlertEvaluator:
    """
    Evaluates a PipelineResult against configured thresholds.
    Writes alert_log rows for every LOW / EMPTY breach.
    """

    def __init__(self, config: DotMap) -> None:
        alert_cfg = getattr(config, "alerts", DotMap())
        self._enabled: bool = bool(getattr(alert_cfg, "enabled", False))

        thresholds_cfg = getattr(alert_cfg, "thresholds", DotMap())

        water_t = getattr(thresholds_cfg, "water", DotMap())
        self._water_low: float = float(water_t.get("low", 20.0))
        self._water_empty: float = float(water_t.get("empty", 5.0))

        food_t = getattr(thresholds_cfg, "food", DotMap())
        self._food_low: float = float(food_t.get("low", 20.0))
        self._food_empty: float = float(food_t.get("empty", 5.0))

        logger.info(
            f"AlertEvaluator initialised | enabled={self._enabled} "
            f"| water low={self._water_low} empty={self._water_empty} "
            f"| food low={self._food_low} empty={self._food_empty}"
        )

    async def evaluate(
        self,
        session: AsyncSession,
        result: PipelineResult,
        cage_id: str,
    ) -> List[AlertRecord]:
        """
        Evaluate thresholds, write alert_log rows, return fired alerts.
        Returns [] when alerts disabled or session is None.
        """
        if not self._enabled or session is None:
            return []

        fired: List[AlertRecord] = []

        # ── water ─────────────────────────────────────────────────────────
        if (
            result.water_pct is not None
            and result.water_label not in (_UNDETERMINED, "MISSING")
        ):
            alert = await self._check_level(
                session=session,
                target="water",
                value=result.water_pct,
                low_threshold=self._water_low,
                empty_threshold=self._water_empty,
                cage_id=cage_id,
            )
            if alert:
                fired.append(alert)

        # ── food ──────────────────────────────────────────────────────────
        if (
            result.food_pct is not None
            and result.food_label not in (_UNDETERMINED,)
        ):
            alert = await self._check_level(
                session=session,
                target="food",
                value=result.food_pct,
                low_threshold=self._food_low,
                empty_threshold=self._food_empty,
                cage_id=cage_id,
            )
            if alert:
                fired.append(alert)

        return fired

    async def _check_level(
        self,
        session: AsyncSession,
        target: str,
        value: float,
        low_threshold: float,
        empty_threshold: float,
        cage_id: str,
    ) -> Optional[AlertRecord]:
        """
        Check one level value against EMPTY then LOW threshold.
        Returns AlertRecord if breached, else None.
        """
        if value <= empty_threshold:
            alert_type = "EMPTY"
        elif value <= low_threshold:
            alert_type = "LOW"
        else:
            return None

        message = (
            f"{target.capitalize()} level {alert_type} | "
            f"cage={cage_id} | value={value:.1f}%"
        )
        alert = AlertRecord(
            target=target,
            alert_type=alert_type,
            value=value,
            message=message,
        )

        await insert_alert(session=session, alert=alert, cage_id=cage_id)
        logger.info(
            f"Alert fired | target={target} type={alert_type} "
            f"value={value:.1f}% cage={cage_id}"
        )
        return alert
