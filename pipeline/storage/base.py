"""
pipeline/storage/base.py

Abstract base classes for all storage and image store implementations.

FIX: Added two abstract methods required by the voting system in cooldown.py:
     - count_recent_threshold_breaches(target, minutes) → int
     - record_threshold_breach(target) → None
     These were called in cooldown.py but never defined here or in postgres.py,
     causing an AttributeError at runtime whenever require_consecutive_alerts > 1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from dotmap import DotMap

from core.result import AlertRecord, PipelineResult


class BaseStorage(ABC):
    """Abstract storage — all DB backends must subclass this."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Establish the database connection / pool."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the database connection / pool."""
        ...

    @abstractmethod
    async def save_result(self, result: PipelineResult) -> str:
        """
        Persist a PipelineResult.

        Returns:
            The assigned UUID string for the new record.
        """
        ...

    @abstractmethod
    async def update_image_path(self, result_id: str, image_path: str) -> None:
        """Update the image_path field after image has been stored."""
        ...

    @abstractmethod
    async def get_results(
        self,
        limit: int = 20,
        offset: int = 0,
        target: Optional[str] = None,
    ) -> List[dict]:
        """
        Retrieve past pipeline results ordered by processed_at DESC.

        Args:
            limit:  Maximum number of records to return.
            offset: Pagination offset.
            target: If provided, filter to results where this target was measured.
        """
        ...

    @abstractmethod
    async def save_alert(self, alert: AlertRecord) -> None:
        """Persist an alert record to alert_log."""
        ...

    @abstractmethod
    async def get_last_alert_at(self, target: str) -> Optional[str]:
        """
        Return the ISO timestamp of the last alert for `target`,
        or None if no alert has been fired.
        """
        ...

    @abstractmethod
    async def upsert_cooldown(self, target: str) -> None:
        """Update (or insert) the cooldown_state row for `target` to now()."""
        ...

    # ------------------------------------------------------------------
    # FIX 3: Voting system methods — were called in cooldown.py but
    # never defined in base or postgres, causing AttributeError at runtime.
    # ------------------------------------------------------------------

    @abstractmethod
    async def record_threshold_breach(self, target: str) -> None:
        """
        Record a threshold breach event for `target`.

        Called by the voting system every time a target's value is below
        threshold. Used to count consecutive breaches within a time window.
        """
        ...

    @abstractmethod
    async def count_recent_threshold_breaches(
        self, target: str, minutes: int
    ) -> int:
        """
        Return the count of threshold breach events for `target`
        recorded within the last `minutes` minutes.

        Used by the voting system to require N consecutive breaches
        before firing an alert.
        """
        ...


class BaseImageStore(ABC):
    """Abstract image store — saves raw image bytes to a backing store."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    async def save(self, image_bytes: bytes, filename: str) -> str:
        """
        Persist image bytes.

        Args:
            image_bytes: Raw image data.
            filename:    Suggested filename (may be modified by implementation).

        Returns:
            The canonical path or URL where the image is stored.
        """
        ...

    @abstractmethod
    async def get_url(self, stored_path: str) -> str:
        """Return a publicly accessible URL (or local path) for the stored image."""
        ...