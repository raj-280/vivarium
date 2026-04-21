"""
pipeline/storage/base.py

Abstract base classes for database storage and image store.
All concrete implementations must subclass these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from dotmap import DotMap

from core.result import AlertRecord, PipelineResult


class BaseStorage(ABC):
    """Abstract database storage — persists pipeline results and alerts."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    async def connect(self) -> None:
        """Establish database connection / pool."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection / pool."""
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
