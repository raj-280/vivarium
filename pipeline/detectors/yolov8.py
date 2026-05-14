"""
pipeline/detectors/yolov8.py

YOLOv8 closed-vocabulary detector implementation.
Drop-in replacement for YOLOv8WorldDetector once you have a fine-tuned
YOLOv8s checkpoint trained on your vivarium images.

Config block required in config.yaml:
    detector:
      engine: yolov8
      model_path: ./models/weights/yolov8s_vivarium.pt  # your trained weights
      device: cpu
      min_confidence: 0.45
      yolov8:
        class_map:
          water: 0
          food: 1
          mouse: 2
        nms_iou: 0.45
        agnostic_nms: true

Training instructions (run once you have labeled data):
    yolo detect train \
        model=yolov8s.pt \
        data=vivarium.yaml \
        epochs=100 \
        imgsz=640 \
        device=0

Dataset yaml (vivarium.yaml) format:
    path: ./data/vivarium
    train: images/train
    val: images/val
    names:
      0: water
      1: food
      2: mouse
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox
from pipeline.detectors.base import BaseDetector


class YOLOv8Detector(BaseDetector):
    """Closed-vocabulary detector backed by a fine-tuned YOLOv8s checkpoint."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._model = None
        self._class_map: Dict[int, str] = {}   # idx → target name
        self._target_to_idx: Dict[str, int] = {}  # target name → idx

    def load(self) -> None:
        """Load YOLOv8 model weights from config.detector.model_path."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        model_path = self.config.detector.model_path
        device = self.config.detector.device
        yolov8_cfg = self.config.detector.yolov8

        # Build class maps from config
        raw_class_map = {
            k: v for k, v in yolov8_cfg.class_map.items()
        }
        self._class_map = {int(v): str(k) for k, v in raw_class_map.items()}
        self._target_to_idx = {str(k): int(v) for k, v in raw_class_map.items()}

        logger.info(
            f"Loading YOLOv8 model from '{model_path}' on device={device} "
            f"| class_map={self._class_map}"
        )
        self._model = YOLO(model_path)
        self._model.to(device)
        logger.info("YOLOv8 model loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run closed-vocabulary detection for each target.

        Args:
            image:   Preprocessed BGR numpy array (H x W x 3).
            targets: List of target names (e.g. ["water", "food", "mouse"]).

        Returns:
            Dict mapping target → BoundingBox (highest-confidence match) or None.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded — call .load() first")

        yolov8_cfg = self.config.detector.yolov8
        min_conf = float(self.config.detector.min_confidence)
        nms_iou = float(getattr(yolov8_cfg, "nms_iou", 0.45))
        agnostic_nms = bool(getattr(yolov8_cfg, "agnostic_nms", True))

        h, w = image.shape[:2]

        predictions = self._model.predict(
            source=image,
            device=self.config.detector.device,
            conf=min_conf,
            iou=nms_iou,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )

        # Only keep detections for requested targets
        valid_indices = {
            self._target_to_idx[t] for t in targets if t in self._target_to_idx
        }

        # For each target keep the highest-confidence detection
        best: Dict[str, Optional[BoundingBox]] = {t: None for t in targets}

        for pred in predictions:
            if pred.boxes is None:
                continue
            for box in pred.boxes:
                cls_idx = int(box.cls[0].item())
                if cls_idx not in valid_indices:
                    continue

                target = self._class_map.get(cls_idx)
                if target is None or target not in targets:
                    continue

                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()  # absolute pixels [x1, y1, x2, y2]

                bbox = BoundingBox(
                    x1=xyxy[0] / w,
                    y1=xyxy[1] / h,
                    x2=xyxy[2] / w,
                    y2=xyxy[3] / h,
                    confidence=conf,
                    label=target,
                )

                if best[target] is None or conf > best[target].confidence:
                    best[target] = bbox

        for target, bbox in best.items():
            if bbox is not None:
                logger.debug(
                    f"Detected '{target}' | conf={bbox.confidence:.3f} "
                    f"| box=[{bbox.x1:.2f},{bbox.y1:.2f},"
                    f"{bbox.x2:.2f},{bbox.y2:.2f}]"
                )
            else:
                logger.debug(f"No detection for target '{target}'")

        return best