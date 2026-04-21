"""
pipeline/threshold/cooldown.py

Per-target cooldown logic with voting system.
Reads last_alert_at from DB and determines whether enough time has passed
since the last notification before allowing a new one to fire.
Also implements a voting system: require consecutive threshold breaches
before firing an alert.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from dotmap import DotMap
from loguru import logger

from pipeline.storage.base import BaseStorage


class CooldownManager:
    """Checks and updates per-target notification cooldowns with voting."""

    def __init__(self, config: DotMap, storage: BaseStorage) -> None:
        self.config = config
        self._storage = storage
        self._cooldown_minutes: int = int(config.notifiers.cooldown_minutes)
        self._require_consecutive: int = int(getattr(config.notifiers, 'require_consecutive_alerts', 1))
        self._voting_window_minutes: int = int(getattr(config.notifiers, 'voting_window_minutes', 5))

    async def is_cooled_down(self, target: str) -> bool:
        """
        Return True if the cooldown period has elapsed for `target`
        (i.e., a new notification is allowed).

        Returns False if the cooldown has NOT expired yet.
        """
        last_at_str: Optional[str] = await self._storage.get_last_alert_at(target)

        if last_at_str is None:
            logger.debug(f"Cooldown check [{target}]: no previous alert — allowed")
            return True

        try:
            last_at = datetime.fromisoformat(last_at_str)
        except ValueError:
            logger.warning(f"Could not parse last_alert_at '{last_at_str}' for '{target}'")
            return True

        # Ensure timezone-aware comparison
        if last_at.tzinfo is None:
            last_at = last_at.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        elapsed = now - last_at
        cooldown_delta = timedelta(minutes=self._cooldown_minutes)

        if elapsed >= cooldown_delta:
            logger.debug(
                f"Cooldown check [{target}]: elapsed={elapsed} >= "
                f"{cooldown_delta} — allowed"
            )
            return True
        else:
            remaining = cooldown_delta - elapsed
            logger.info(
                f"Cooldown active [{target}]: {remaining} remaining — skipping notification"
            )
            return False

    async def should_fire_with_voting(self, target: str) -> bool:
        """
        **OPTIMIZATION: Voting system**
        
        Return True if:
          1. Cooldown has expired, AND
          2. We've seen >= require_consecutive_alerts breaches in the voting window.
        
        Returns False if voting hasn't reached threshold yet.
        """
        # Check basic cooldown first
        cooled = await self.is_cooled_down(target)
        if not cooled:
            logger.debug(f"Voting system [{target}]: cooldown not expired")
            return False

        # Count consecutive breach records within voting window
        consecutive_count = await self._storage.count_recent_threshold_breaches(
            target,
            minutes=self._voting_window_minutes,
        )

        if consecutive_count >= self._require_consecutive:
            logger.info(
                f"Voting system [{target}]: {consecutive_count} breaches >= "
                f"{self._require_consecutive} required — FIRE alert"
            )
            return True
        else:
            logger.debug(
                f"Voting system [{target}]: {consecutive_count} breaches < "
                f"{self._require_consecutive} required — vote pending"
            )
            return False

    async def mark_alerted(self, target: str) -> None:
        """Record that an alert was fired for `target` right now."""
        await self._storage.upsert_cooldown(target)
        logger.debug(f"Cooldown timestamp updated for target={target}")

    async def record_breach(self, target: str) -> None:
        """**OPTIMIZATION: Record a threshold breach for voting system."""
        await self._storage.record_threshold_breach(target)
        logger.debug(f"Threshold breach recorded for target={target} (voting system)")
