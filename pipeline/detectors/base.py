"""
pipeline/detectors/base.py

Abstract base class for all detector implementations.
Every detector receives the full preprocessed image (numpy array) and
returns a mapping of target name → BoundingBox (or None if not found).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from dotmap import DotMap

from core.result import BoundingBox


class BaseDetector(ABC):
    """Abstract detector — all engines must subclass this."""

    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at startup."""
        ...

    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run detection on a preprocessed image.

        Args:
            image:   Preprocessed BGR numpy array (H x W x 3).
            targets: List of target names to detect (e.g., ["water", "food", "mouse"]).

        Returns:
            Dict mapping each target to a BoundingBox (or None if not detected).
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(engine={self.config.detector.engine})"
