"""
pipeline/measurers/unet_measurer.py

BaseMeasurer wrappers for both U-Net inferencers.
These plug the neural models into the existing measurer factory
so the ComparatorMeasurer can use them as the ml_engine alongside CV.

Config required:

  water:
    engine: comparator
    comparator:
      cv_engine: opencv_water
      ml_engine: unet_water       # ← this file
      fusion: weighted_average
      disagreement_threshold: 15.0
    unet_water:
      weights_path: models/weights/water_unet.pt
      device: cpu
      mc_passes: 1   # set to 5 for MC dropout confidence (slower)

  food:
    engine: comparator
    comparator:
      cv_engine: opencv_food
      ml_engine: unet_food        # ← this file
      fusion: weighted_average
      disagreement_threshold: 15.0
    unet_food:
      weights_path: models/weights/food_unet.pt
      device: cpu
      mc_passes: 5   # MC dropout active — 5 forward passes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


class UNetWaterMeasurer(BaseMeasurer):
    """
    ML engine for water level using the trained WaterUNet.
    Plugs into ComparatorMeasurer as ml_engine: unet_water.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._inferencer = None

    def load(self) -> None:
        from ml_models.water_unet import WaterUNetInferencer

        target_cfg = getattr(self.config.measurers, self.target)
        unet_cfg = target_cfg.unet_water
        weights_path = Path(unet_cfg.weights_path)
        device = str(getattr(unet_cfg, "device", "cpu"))

        if not weights_path.exists():
            raise FileNotFoundError(
                f"WaterUNet weights not found at '{weights_path}'. "
                f"Train the model first using WaterUNetTrainer, "
                f"or set water.unet_water.weights_path in config.yaml."
            )

        logger.info(f"Loading WaterUNet weights from {weights_path} on {device}")
        self._inferencer = WaterUNetInferencer.from_weights(weights_path, device=device)
        self._tube_top_y = int(getattr(unet_cfg, "tube_top_y", 0))
        logger.info(f"WaterUNet loaded successfully (tube_top_y={self._tube_top_y})")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        if self._inferencer is None:
            raise RuntimeError("UNetWaterMeasurer not loaded — call .load() first")

        mask, water_pct, confidence = self._inferencer.predict(roi, tube_top_y=self._tube_top_y)

        label = f"UNet water: {water_pct:.1f}% | conf={confidence:.2f}"
        logger.debug(label)

        return MeasurementResult(
            level=water_pct,
            confidence=confidence,
            label=label,
            present=None,
        )


class UNetFoodMeasurer(BaseMeasurer):
    """
    ML engine for food level using the trained FoodUNet.
    Plugs into ComparatorMeasurer as ml_engine: unet_food.
    """

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._inferencer = None

    def load(self) -> None:
        from ml_models.food_unet import FoodUNetInferencer

        target_cfg = getattr(self.config.measurers, self.target)
        unet_cfg = target_cfg.unet_food
        weights_path = Path(unet_cfg.weights_path)
        device = str(getattr(unet_cfg, "device", "cpu"))
        self._mc_passes = int(getattr(unet_cfg, "mc_passes", 1))

        if not weights_path.exists():
            raise FileNotFoundError(
                f"FoodUNet weights not found at '{weights_path}'. "
                f"Train the model first using FoodUNetTrainer, "
                f"or set food.unet_food.weights_path in config.yaml."
            )

        logger.info(f"Loading FoodUNet weights from {weights_path} on {device}")
        self._inferencer = FoodUNetInferencer.from_weights(weights_path, device=device)
        logger.info(f"FoodUNet loaded successfully (MC passes={self._mc_passes})")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        if self._inferencer is None:
            raise RuntimeError("UNetFoodMeasurer not loaded — call .load() first")

        surface_y, food_pct, confidence = self._inferencer.predict(
            roi, n_passes=self._mc_passes
        )

        label = f"UNet food: {food_pct:.1f}% | surface_y={surface_y} | conf={confidence:.2f}"
        logger.debug(label)

        return MeasurementResult(
            level=food_pct,
            confidence=confidence,
            label=label,
            present=None,
        )
