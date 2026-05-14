"""
pipeline/detectors/yolov8world.py

YOLOv8-World detector implementation.
Uses ultralytics YOLOWorld model for open-vocabulary object detection.
All prompts and thresholds are read from config — zero hardcoded values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox
from pipeline.detectors.base import BaseDetector


class YOLOv8WorldDetector(BaseDetector):
    """Open-vocabulary detector backed by YOLOv8-World (ultralytics)."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._model = None

    def load(self) -> None:
        """Load YOLOWorld model weights. Downloads if not present."""
        try:
            from ultralytics import YOLOWorld  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        model_path = self.config.detector.model_path
        device = self.config.detector.device

        logger.info(f"Loading YOLOv8-World model from {model_path} on device={device}")
        self._model = YOLOWorld(model_path)
        self._model.to(device)
        logger.info("YOLOv8-World model loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run open-vocabulary detection for each target using its configured prompt.

        Args:
            image:   Preprocessed BGR numpy array (H x W x 3).
            targets: List of target names (e.g., ["water", "food", "mouse"]).

        Returns:
            Dict mapping target → BoundingBox (highest-confidence match) or None.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded — call .load() first")

        prompts_cfg = self.config.detector.prompts
        results: Dict[str, Optional[BoundingBox]] = {}

        # Build prompt list for all targets in one forward pass
        prompt_list = [getattr(prompts_cfg, t) for t in targets]
        self._model.set_classes(prompt_list)

        h, w = image.shape[:2]
        predictions = self._model.predict(
            source=image,
            device=self.config.detector.device,
            verbose=False,
        )

        # Map class index → target name
        idx_to_target = {i: t for i, t in enumerate(targets)}

        # For each target, keep the highest-confidence detection
        best: Dict[str, Optional[BoundingBox]] = {t: None for t in targets}

        for pred in predictions:
            if pred.boxes is None:
                continue
            for box in pred.boxes:
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                target = idx_to_target.get(cls_idx)
                if target is None:
                    continue

                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2] absolute pixels
                bbox = BoundingBox(
                    x1=xyxy[0] / w,
                    y1=xyxy[1] / h,
                    x2=xyxy[2] / w,
                    y2=xyxy[3] / h,
                    confidence=conf,
                    label=getattr(prompts_cfg, target),
                )

                if best[target] is None or conf > best[target].confidence:
                    best[target] = bbox

        for target, bbox in best.items():
            if bbox is not None:
                logger.debug(
                    f"Detected '{target}' | conf={bbox.confidence:.3f} "
                    f"| box=[{bbox.x1:.2f},{bbox.y1:.2f},{bbox.x2:.2f},{bbox.y2:.2f}]"
                )
            else:
                logger.debug(f"No detection for target '{target}'")
            results[target] = bbox

        return results
