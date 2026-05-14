"""
pipeline/measurers/base.py

Abstract base class for all measurer implementations.
Each measurer receives a cropped ROI (numpy array) of a detected target
and returns a MeasurementResult with level, confidence, and label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dotmap import DotMap

from core.result import MeasurementResult


class BaseMeasurer(ABC):
    """Abstract measurer — all engines must subclass this."""

    def __init__(self, config: DotMap, target: str) -> None:
        self.config = config
        self.target = target  # "water" | "food" | "mouse"

    @abstractmethod
    def load(self) -> None:
        """Load model / assets. Called once at startup."""
        ...

    @abstractmethod
    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Estimate the level of the target from the cropped ROI image.

        Args:
            roi: Cropped BGR numpy array for the target region.

        Returns:
            MeasurementResult with level (0–100), confidence (0–1), label.
        """
        ...

    def __repr__(self) -> str:
        target_cfg = getattr(self.config.measurers, self.target)
        return f"{self.__class__.__name__}(target={self.target}, engine={target_cfg.engine})"
