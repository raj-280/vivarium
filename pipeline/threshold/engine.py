"""
pipeline/threshold/engine.py

Threshold engine — compares pipeline results to configured thresholds,
applies cooldown logic, renders alert messages, and fires notifiers.

All threshold values and message templates are read from config.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from dotmap import DotMap
from loguru import logger

from core.result import AlertRecord, PipelineResult
from pipeline.notifiers.base import BaseNotifier
from pipeline.storage.base import BaseStorage
from pipeline.threshold.cooldown import CooldownManager


class ThresholdEngine:
    """Evaluates pipeline results against configured thresholds."""

    def __init__(
        self,
        config: DotMap,
        storage: BaseStorage,
        notifiers: List[BaseNotifier],
    ) -> None:
        self.config = config
        self._storage = storage
        self._notifiers = notifiers
        self._cooldown = CooldownManager(config, storage)

    async def evaluate(self, result: PipelineResult) -> List[AlertRecord]:
        """
        Evaluate a PipelineResult against all configured thresholds.

        For each threshold breach:
          1. Check cooldown via DB.
          2. Render message from config template.
          3. Fire all enabled notifiers.
          4. Persist alert to DB and update cooldown timestamp.

        Returns:
            List of AlertRecord objects for alerts that were fired.
        """
        fired: List[AlertRecord] = []
        thresholds = self.config.thresholds
        templates = self.config.notifiers.templates
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        # --- Water level check ---
        if result.water_pct is not None:
            water_threshold = float(thresholds.water_low_pct)
            if result.water_pct < water_threshold:
                alert = await self._maybe_fire(
                    target="water",
                    alert_type="water_low",
                    template=templates.water_low,
                    value=result.water_pct,
                    threshold=water_threshold,
                    now_iso=now_iso,
                )
                if alert:
                    fired.append(alert)

        # --- Food level check ---
        if result.food_pct is not None:
            food_threshold = float(thresholds.food_low_pct)
            if result.food_pct < food_threshold:
                alert = await self._maybe_fire(
                    target="food",
                    alert_type="food_low",
                    template=templates.food_low,
                    value=result.food_pct,
                    threshold=food_threshold,
                    now_iso=now_iso,
                )
                if alert:
                    fired.append(alert)

        # --- Mouse missing check ---
        # Mouse missing is handled differently — it is time-based, not percentage-based.
        # We fire if mouse_present is False.
        if result.mouse_present is False:
            mouse_missing_minutes = int(thresholds.mouse_missing_minutes)
            alert = await self._maybe_fire(
                target="mouse",
                alert_type="mouse_missing",
                template=templates.mouse_missing,
                value=None,
                threshold=None,
                now_iso=now_iso,
                extra_minutes=mouse_missing_minutes,
            )
            if alert:
                fired.append(alert)

        return fired

    async def fire_image_rejected(self, reason: str) -> AlertRecord | None:
        """Fire an image_rejected alert."""
        templates = self.config.notifiers.templates
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        return await self._maybe_fire(
            target="image",
            alert_type="image_rejected",
            template=templates.image_rejected,
            value=None,
            threshold=None,
            now_iso=now_iso,
            reason=reason,
        )

    async def _maybe_fire(
        self,
        target: str,
        alert_type: str,
        template: str,
        value,
        threshold,
        now_iso: str,
        extra_minutes: int | None = None,
        reason: str | None = None,
    ) -> AlertRecord | None:
        """Check cooldown, render message, fire notifiers, persist record."""
        cooled = await self._cooldown.is_cooled_down(target)
        if not cooled:
            return None

        # Render template — replace all known placeholders
        message = str(template)
        if value is not None:
            message = message.replace("{value}", f"{value:.1f}")
        if threshold is not None:
            message = message.replace("{threshold}", f"{threshold:.1f}")
        if extra_minutes is not None:
            message = message.replace("{minutes}", str(extra_minutes))
        message = message.replace("{timestamp}", now_iso)
        if reason is not None:
            message = message.replace("{reason}", reason)

        logger.info(f"Threshold breached | target={target} | alert_type={alert_type} | msg={message}")

        # Fire all notifiers
        notifiers_fired: List[str] = []
        for notifier in self._notifiers:
            try:
                success = await notifier.send(message, alert_type)
                if success:
                    notifiers_fired.append(notifier.__class__.__name__)
            except Exception as exc:
                logger.error(f"Notifier {notifier} failed: {exc}")

        # Persist alert record
        alert = AlertRecord(
            target=target,
            alert_type=alert_type,
            value=value,
            message=message,
            notifiers_fired=notifiers_fired,
        )
        await self._storage.save_alert(alert)
        await self._cooldown.mark_alerted(target)

        return alert
