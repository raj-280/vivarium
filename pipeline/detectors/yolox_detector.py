"""
pipeline/detectors/yolox_detector.py

YOLOX closed-vocabulary detector for the Vivarium pipeline.

Trained on 13 classes (matches yolox_vivarium_tiny.py exp file):
    0   mouse
    1   water_critical   (0–15%)
    2   water_low        (15–35%)
    3   water_ok         (35–80%)
    4   water_full       (80–100%)
    5   food_critical    (0–15%)
    6   food_low         (15–35%)
    7   food_ok          (35–80%)
    8   food_full        (80–100%)
    9   bedding_worst
    10  bedding_bad
    11  bedding_ok
    12  bedding_perfect

detect() groups the 13 classes into 4 logical targets:
    "mouse"   → class 0         — returns bbox + label="mouse_cls0"
    "water"   → classes 1–4     — returns highest-conf bbox + label="water_cls<N>"
    "food"    → classes 5–8     — returns highest-conf bbox + label="food_cls<N>"
    "bedding" → classes 9–12    — returns highest-conf bbox + label="bedding_cls<N>"

The class ID is embedded in the label so YOLOXMeasurer can decode
the level / condition without needing any extra model.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox
from pipeline.detectors.base import BaseDetector

# ── Class ID groupings ────────────────────────────────────────────────────────
_MOUSE_CLS = {0}
_WATER_CLS = {1, 2, 3, 4}
_FOOD_CLS  = {5, 6, 7, 8}
_BED_CLS   = {9, 10, 11, 12}

_TARGET_TO_GROUP: Dict[str, set] = {
    "mouse":   _MOUSE_CLS,
    "water":   _WATER_CLS,
    "food":    _FOOD_CLS,
    "bedding": _BED_CLS,
}


class YOLOXDetector(BaseDetector):
    """YOLOX detector — returns one BoundingBox per logical target."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._model = None
        self._preproc = None
        self._input_size: tuple[int, int] = (416, 416)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load YOLOX weights from config.detector.model_path."""
        try:
            from yolox.exp import get_exp
            from yolox.data.data_augment import ValTransform
        except ImportError as exc:
            raise ImportError(
                "yolox is not installed. "
                "Run: pip install yolox  (or install from source)"
            ) from exc

        cfg = self.config.detector
        yolox_cfg = cfg.yolox

        model_path  = str(cfg.model_path)
        device      = str(cfg.device)
        exp_file    = str(yolox_cfg.exp_file)
        num_classes = int(yolox_cfg.num_classes)

        raw_size = yolox_cfg.input_size          # list [h, w] from yaml
        self._input_size = (int(raw_size[0]), int(raw_size[1]))

        logger.info(
            f"[YOLOXDetector] Loading exp from {exp_file} | "
            f"weights={model_path} | device={device} | "
            f"num_classes={num_classes} | input_size={self._input_size}"
        )

        exp = get_exp(exp_file, exp_name=None)
        exp.num_classes = num_classes

        self._model = exp.get_model()
        self._model.eval()

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model.load_state_dict(ckpt.get("model", ckpt))
        self._model.to(device)

        self._preproc = ValTransform(legacy=False)
        self._device  = device

        logger.info("[YOLOXDetector] Model loaded successfully")

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run YOLOX inference and return one BoundingBox per requested target.

        The class ID is embedded in bbox.label as  "<target>_cls<N>"
        so YOLOXMeasurer can decode level/condition without a second model.

        Args:
            image:   Preprocessed BGR numpy array (H x W x 3).
            targets: e.g. ["water", "food", "mouse", "bedding"]

        Returns:
            Dict mapping each target → BoundingBox (or None if not detected).
        """
        if self._model is None:
            raise RuntimeError("YOLOXDetector not loaded — call .load() first")

        cfg        = self.config.detector
        min_conf   = float(cfg.min_confidence)
        nms_iou    = float(cfg.yolox.nms_iou)
        num_classes = int(cfg.yolox.num_classes)

        orig_h, orig_w = image.shape[:2]

        # ── Preprocess ────────────────────────────────────────────────
        img_tensor, _ = self._preproc(image, None, self._input_size)
        # Compute ratio ourselves — ValTransform's returned ratio is unreliable
        ratio = min(self._input_size[0] / orig_h, self._input_size[1] / orig_w)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()
        if self._device != "cpu":
            img_tensor = img_tensor.cuda()

        # ── Inference ────────────────────────────────────────────────
        with torch.no_grad():
            raw = self._model(img_tensor)

        from yolox.utils import postprocess
        outputs = postprocess(
            raw,
            num_classes=num_classes,
            conf_thre=min_conf,
            nms_thre=nms_iou,
        )

        # ── Parse raw detections ──────────────────────────────────────
        result: Dict[str, Optional[BoundingBox]] = {t: None for t in targets}

        if outputs[0] is None:
            logger.debug("[YOLOXDetector] No detections in frame")
            return result

        detections = outputs[0].cpu().numpy()
        # columns: x1, y1, x2, y2, obj_conf, cls_conf, cls_id

        # Undo letterbox scaling so coords are in original image space
        # ratio already normalised to scalar above

        pad_left = (self._input_size[1] - orig_w * ratio) / 2
        pad_top  = (self._input_size[0] - orig_h * ratio) / 2

        for det in detections:
            x1_l, y1_l, x2_l, y2_l = det[0], det[1], det[2], det[3]
            obj_conf  = float(det[4])
            cls_conf  = float(det[5])
            cls_id    = int(det[6])
            score     = obj_conf * cls_conf

            if score < min_conf:
                continue

            # Convert from letterbox space → original image pixels
            x1 = np.clip((x1_l - pad_left) / ratio, 0, orig_w)
            y1 = np.clip((y1_l - pad_top)  / ratio, 0, orig_h)
            x2 = np.clip((x2_l - pad_left) / ratio, 0, orig_w)
            y2 = np.clip((y2_l - pad_top)  / ratio, 0, orig_h)

            # Normalise to 0–1 (matches BoundingBox convention)
            x1n = float(x1 / orig_w)
            y1n = float(y1 / orig_h)
            x2n = float(x2 / orig_w)
            y2n = float(y2 / orig_h)

            # Find which logical target this class belongs to
            matched_target = None
            for target in targets:
                if cls_id in _TARGET_TO_GROUP.get(target, set()):
                    matched_target = target
                    break

            if matched_target is None:
                continue

            # Keep highest-confidence detection per target
            current = result[matched_target]
            if current is None or score > current.confidence:
                result[matched_target] = BoundingBox(
                    x1=x1n,
                    y1=y1n,
                    x2=x2n,
                    y2=y2n,
                    confidence=score,
                    # Embed class ID so measurer can decode level/condition
                    label=f"{matched_target}_cls{cls_id}",
                )

        for target, bbox in result.items():
            if bbox is not None:
                logger.debug(
                    f"[YOLOXDetector] {target} | cls_label={bbox.label} "
                    f"conf={bbox.confidence:.3f} "
                    f"box=[{bbox.x1:.3f},{bbox.y1:.3f},{bbox.x2:.3f},{bbox.y2:.3f}]"
                )
            else:
                logger.debug(f"[YOLOXDetector] {target} → not detected")

        return result
