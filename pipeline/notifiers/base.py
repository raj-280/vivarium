"""
pipeline/notifiers/base.py

Abstract base class for all notifier implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from dotmap import DotMap


class BaseNotifier(ABC):
    """Abstract notifier — all implementations must subclass this."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    async def send(self, message: str, alert_type: str) -> bool:
        """
        Send a notification message.

        Args:
            message:    Rendered notification text (placeholders already filled).
            alert_type: Type of alert (water_low | food_low | mouse_missing | image_rejected).

        Returns:
            True if sent successfully, False otherwise.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
