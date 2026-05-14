"""
pipeline/measurers/pspnet_measurer.py

BaseMeasurer wrapper around PSPNet (ResNet50 + PSP) for water level estimation.
Registered in MeasurerFactory as engine key: 'fcn_psp'

Pipeline:
    1. Preprocess ROI → normalised tensor
    2. Run PSPNet → liquid probability mask
    3. Compute fill % from liquid mask height ratio

Config block (config.yaml):
    water:
      engine: fcn_psp
      model_path: models/weights/water_pspnet.pt
      device: cpu
      input_size: 473          # square input size for PSPNet (473 or 512)
      min_liquid_confidence: 0.5  # pixel threshold for liquid mask
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer
from ml_models.pspnet_model import PSPNet


class PSPNetWaterMeasurer(BaseMeasurer):
    """
    Wraps PSPNet (ResNet50 + PSP) into the BaseMeasurer interface.
    Performs pixel-level liquid segmentation and computes fill %.
    """

    # ImageNet normalisation — same as LabPics training
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._model: PSPNet | None = None
        self._device: str = "cpu"
        self._input_size: int = 473
        self._threshold: float = 0.5

    def load(self) -> None:
        """Load PSPNet weights from config path."""
        target_cfg = getattr(self.config.measurers, self.target)

        model_path_str: str = getattr(target_cfg, "model_path", "")
        self._device = str(getattr(target_cfg, "device", "cpu"))
        self._input_size = int(getattr(target_cfg, "input_size", 473))
        self._threshold = float(getattr(target_cfg, "min_liquid_confidence", 0.5))

        if not model_path_str or model_path_str in ("null", "None", ""):
            raise ValueError(
                f"fcn_psp engine selected for '{self.target}' but "
                f"{self.target}.model_path is not set in config.yaml."
            )

        model_path = Path(model_path_str)
        if not model_path.exists():
            raise FileNotFoundError(
                f"PSPNet weights not found at '{model_path}'. "
                f"Check config.yaml → {self.target}.model_path"
            )

        logger.info(
            f"[PSPNetWaterMeasurer] Loading PSPNet from {model_path} "
            f"on device={self._device}"
        )

        # Construct with no pretrained weights — our checkpoint replaces everything
        self._model = PSPNet(num_classes=2)
        state = torch.load(str(model_path), map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

        if self._device != "cpu":
            self._model = self._model.to(self._device)

        logger.info("[PSPNetWaterMeasurer] PSPNet loaded successfully")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Run PSPNet on the water bottle ROI and return fill %.

        Args:
            roi: BGR numpy array — cropped water bottle region from detector.

        Returns:
            MeasurementResult with level (0–100), confidence (0–1), label.
        """
        if self._model is None:
            raise RuntimeError("PSPNetWaterMeasurer not loaded — call .load() first")

        # --- Preprocess ---
        tensor = self._preprocess(roi)

        # --- Inference ---
        with torch.no_grad():
            logits = self._model(tensor)               # (1, 2, H, W)
            probs = F.softmax(logits, dim=1)           # (1, 2, H, W)
            liquid_prob = probs[0, 1].cpu().numpy()    # (H, W) — liquid channel

        # --- Build binary mask ---
        liquid_mask = (liquid_prob >= self._threshold).astype(np.uint8)

        # --- Compute fill % ---
        level, confidence = self._compute_fill(liquid_mask, liquid_prob)

        label = (
            f"PSPNet [{self.target}]: {level:.1f}% "
            f"(conf={confidence:.3f}, "
            f"liquid_px={liquid_mask.sum()}/{liquid_mask.size})"
        )
        logger.debug(label)

        return MeasurementResult(
            level=round(level, 1),
            confidence=round(confidence, 3),
            label=label,
            present=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, roi: np.ndarray) -> torch.Tensor:
        """
        BGR numpy → normalised float tensor (1, 3, H, W).
        Resizes to self._input_size × self._input_size.
        """
        img = cv2.resize(roi, (self._input_size, self._input_size))
        # BGR → RGB
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        # Normalise with ImageNet stats (same distribution used during training)
        img = (img - self.MEAN) / self.STD
        # HWC → CHW → BCHW
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()

        if self._device != "cpu":
            tensor = tensor.to(self._device)

        return tensor

    def _compute_fill(
        self, mask: np.ndarray, prob: np.ndarray
    ) -> tuple[float, float]:
        """
        Compute fill percentage and confidence from the liquid mask.

        Strategy:
            - Find the topmost row that contains liquid pixels
            - Fill % = liquid_pixels / total_pixels × 100
            - Confidence = mean probability across the detected liquid region

        Returns:
            (level 0–100, confidence 0–1)
        """
        h, w = mask.shape
        row_has_liquid = mask.sum(axis=1) > 0
        liquid_pixel_count = int(mask.sum())

        if liquid_pixel_count == 0:
            return 0.0, 0.3

        logger.debug(
            f"bottle_px (prob>0.05)={(prob > 0.05).sum()} "
            f"| bottle_px (prob>0.1)={(prob > 0.1).sum()} "
            f"| bottle_px (prob>0.2)={(prob > 0.2).sum()} "
            f"| mask.size={mask.size}"
        )

        liquid_rows = np.where(row_has_liquid)[0]
        top_liquid_row = int(liquid_rows.min())
        bottom_liquid_row = int(liquid_rows.max())

        level = float(np.clip(liquid_pixel_count / mask.size * 100.0, 0.0, 100.0))

        # Confidence = mean liquid probability in the detected liquid rows
        region_probs = prob[top_liquid_row:bottom_liquid_row + 1, :]
        confidence = float(np.clip(region_probs.mean(), 0.0, 1.0))

        return level, confidence
