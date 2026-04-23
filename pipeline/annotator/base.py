from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from dotmap import DotMap
from core.result import BoundingBox, MeasurementResult


class BaseAnnotator(ABC):
    def __init__(self, config: DotMap) -> None:
        self.config = config

    @abstractmethod
    def draw(
        self,
        image: np.ndarray,
        gated: Dict[str, Optional[BoundingBox]],
        measurements: Dict[str, MeasurementResult],
        result_id: str,
        filename: str,
    ) -> str:
        """
        Draw boxes + labels on image, save to disk.
        Returns the saved file path.
        """
        ...