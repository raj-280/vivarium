"""
pipeline/detectors/owlvit.py

OWL-ViT (Open-World Localization with Vision Transformer) detector.

Uses HuggingFace `transformers` for zero-shot, text-guided object detection.
Each target is queried by its natural-language prompt from
config.detector.prompts — no class indices, no fine-tuning required.

When to use this over yolov8world:
  - You want a pure-transformer alternative to YOLO for open-vocab detection.
  - YOLOv8-World misses small or oddly-lit objects (OWL-ViT handles patches).
  - You are already in the HuggingFace ecosystem and want consistent tooling.

Supported model IDs (config.detector.model_path):
  google/owlvit-base-patch32      ← fast, good for CPU
  google/owlvit-base-patch16      ← higher spatial resolution
  google/owlvit-large-patch14     ← best accuracy, needs GPU
  google/owlv2-base-patch16       ← OWLv2, improved recall

Config block required (config.yaml → detector section):
    engine: owlvit
    model_path: google/owlvit-base-patch32   # HF hub ID or local directory
    device: cpu
    min_confidence: 0.45
    prompts:                                 # passed verbatim to OWL-ViT
      water: "water bottle with liquid inside"
      food: "food pile in a feeding tray"
      mouse: "small white mouse on white bedding"
    owlvit:
      score_threshold: 0.10    # pre-filter threshold; min_confidence is the
                                # final gate. OWL-ViT raw logits are lower
                                # than YOLO confidences — keep this ≤ 0.15.
      nms_iou: 0.30            # IoU threshold for per-target NMS deduplication
      nms_per_target: true     # run NMS separately per target (recommended)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox
from pipeline.detectors.base import BaseDetector


class OWLViTDetector(BaseDetector):
    """
    Zero-shot detector backed by OWL-ViT / OWLv2 (HuggingFace transformers).

    A single forward pass is run with all target prompts concatenated into one
    batch query. The processor's post_process_object_detection utility converts
    raw logits into (score, label_index, box) triples. Label indices map 1-to-1
    to the `targets` list passed to detect().
    """

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
       from transformers import (
           OwlViTForObjectDetection, OwlViTProcessor,
           Owlv2ForObjectDetection, Owlv2Processor,
       )
   
       model_id: str = self.config.detector.model_path
       device: str = self.config.detector.device
       is_v2 = "owlv2" in model_id.lower()
   
       ModelClass = Owlv2ForObjectDetection if is_v2 else OwlViTForObjectDetection
       ProcessorClass = Owlv2Processor if is_v2 else OwlViTProcessor
   
       logger.info(f"Loading {'OWLv2' if is_v2 else 'OWL-ViT'} '{model_id}' on device='{device}'")
       self._processor = ProcessorClass.from_pretrained(model_id)
       self._model = ModelClass.from_pretrained(model_id)
       self._model.to(device)
       self._model.eval()
       logger.info(f"{'OWLv2' if is_v2 else 'OWL-ViT'} model loaded successfully")
    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run zero-shot text-guided detection for each target.

        A single forward pass covers all targets simultaneously. The processor
        maps each query index back to the corresponding target name, so no
        per-target loop is needed.

        Args:
            image:   Preprocessed BGR numpy array (H × W × 3).
            targets: List of target names whose prompts live in
                     config.detector.prompts (e.g. ["water", "food", "mouse"]).

        Returns:
            Dict mapping each target → highest-confidence BoundingBox, or None.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Detector not loaded — call .load() first")

        try:
            import torch  # type: ignore
            from PIL import Image as PILImage  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "torch and Pillow are required. "
                "Run: pip install torch Pillow"
            ) from exc

        owlvit_cfg = self.config.detector.owlvit
        score_threshold: float = float(getattr(owlvit_cfg, "score_threshold", 0.10))
        nms_iou: float = float(getattr(owlvit_cfg, "nms_iou", 0.30))
        nms_per_target: bool = bool(getattr(owlvit_cfg, "nms_per_target", True))
        min_conf: float = float(self.config.detector.min_confidence)

        prompts_cfg = self.config.detector.prompts
        queries: List[str] = [getattr(prompts_cfg, t) for t in targets]

        # OWL-ViT expects RGB; image array is BGR (OpenCV convention)
        pil_image = PILImage.fromarray(image[..., ::-1])
        h, w = image.shape[:2]
        device: str = self.config.detector.device

        # Processor expects text as a nested list: [[q0, q1, ...]] for batch=1
        inputs = self._processor(
            text=[queries],
            images=pil_image,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=device)
        raw_results = self._processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]  # index 0 = first (only) image in batch

        scores: List[float] = raw_results["scores"].cpu().tolist()
        label_indices: List[int] = raw_results["labels"].cpu().tolist()
        boxes_abs: List[List[float]] = raw_results["boxes"].cpu().tolist()

        # Group raw detections by target for optional per-target NMS
        # Structure: target → list of (score, [x1,y1,x2,y2])
        per_target_raw: Dict[str, List[Tuple[float, List[float]]]] = {
            t: [] for t in targets
        }

        for score, label_idx, box in zip(scores, label_indices, boxes_abs):
            if label_idx >= len(targets):
                continue  # guard against out-of-range indices
            per_target_raw[targets[label_idx]].append((score, box))

        best: Dict[str, Optional[BoundingBox]] = {t: None for t in targets}

        for target, candidates in per_target_raw.items():
            if not candidates:
                continue

            # Sort descending by score; apply greedy NMS if enabled
            candidates.sort(key=lambda x: x[0], reverse=True)
            if nms_per_target and len(candidates) > 1:
                candidates = _greedy_nms(candidates, iou_threshold=nms_iou)

            for score, box in candidates:
                if score < min_conf:
                    continue  # final confidence gate after NMS

                x1, y1, x2, y2 = box
                bbox = BoundingBox(
                    x1=x1 / w,
                    y1=y1 / h,
                    x2=x2 / w,
                    y2=y2 / h,
                    confidence=score,
                    label=getattr(prompts_cfg, target),
                )

                # Take the highest-scoring survivor
                if best[target] is None or score > best[target].confidence:
                    best[target] = bbox
                break  # already sorted; first survivor is best

        for target, bbox in best.items():
            if bbox is not None:
                logger.debug(
                    f"[OWLViT] Detected '{target}' | conf={bbox.confidence:.3f} "
                    f"| box=[{bbox.x1:.2f},{bbox.y1:.2f},"
                    f"{bbox.x2:.2f},{bbox.y2:.2f}]"
                )
            else:
                logger.debug(f"[OWLViT] No detection for target '{target}'")

        return best


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute Intersection-over-Union for two axis-aligned boxes.

    Args:
        box_a: [x1, y1, x2, y2] in absolute pixels.
        box_b: [x1, y1, x2, y2] in absolute pixels.

    Returns:
        IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0.0 else 0.0


def _greedy_nms(
    candidates: List[Tuple[float, List[float]]],
    iou_threshold: float,
) -> List[Tuple[float, List[float]]]:
    """
    Greedy Non-Maximum Suppression on a pre-sorted list of (score, box) pairs.

    Args:
        candidates:    Descending-score list of (score, [x1,y1,x2,y2]).
        iou_threshold: Boxes overlapping above this IoU with a kept box are
                       suppressed.

    Returns:
        Filtered list of survivors, preserving original order.
    """
    kept: List[Tuple[float, List[float]]] = []
    for score, box in candidates:
        suppressed = any(
            _iou(box, kept_box) > iou_threshold for _, kept_box in kept
        )
        if not suppressed:
            kept.append((score, box))
    return kept