"""
pipeline/detectors/groundingdino.py

Grounding DINO detector implementation.
Uses transformers-based GroundingDINO for open-vocabulary grounded detection.
All prompts and thresholds are read from config — zero hardcoded values.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger
from PIL import Image as PILImage

from core.result import BoundingBox
from pipeline.detectors.base import BaseDetector


class GroundingDINODetector(BaseDetector):
    """Open-vocabulary detector backed by Grounding DINO (transformers)."""

    def __init__(self, config: DotMap) -> None:
        super().__init__(config)
        self._processor = None
        self._model = None

    def load(self) -> None:
        """Load Grounding DINO processor and model weights."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are not installed. "
                "Run: pip install transformers torch"
            ) from exc

        model_path = self.config.detector.model_path or "IDEA-Research/grounding-dino-base"
        device = self.config.detector.device

        logger.info(f"Loading Grounding DINO from {model_path} on device={device}")
        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        self._device = device
        self._model.to(device)
        self._model.eval()
        logger.info("Grounding DINO model loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        targets: list[str],
    ) -> Dict[str, Optional[BoundingBox]]:
        """
        Run Grounding DINO detection for each target using its configured prompt.

        Args:
            image:   Preprocessed BGR numpy array (H x W x 3).
            targets: List of target names (e.g., ["water", "food", "mouse"]).

        Returns:
            Dict mapping target → BoundingBox (highest-confidence match) or None.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Detector not loaded — call .load() first")

        import torch  # type: ignore

        prompts_cfg = self.config.detector.prompts
        min_conf: float = self.config.detector.min_confidence

        # Convert BGR numpy array to PIL RGB image
        pil_image = PILImage.fromarray(image[:, :, ::-1])
        h, w = image.shape[:2]

        results: Dict[str, Optional[BoundingBox]] = {t: None for t in targets}

        for target in targets:
            prompt_text: str = getattr(prompts_cfg, target)
            # DINO expects text ending with "."
            text_prompt = prompt_text if prompt_text.endswith(".") else prompt_text + "."

            inputs = self._processor(
                images=pil_image,
                text=text_prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process to get boxes + scores
            target_sizes = torch.tensor([[h, w]], device=self._device)
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=min_conf,
                text_threshold=min_conf,
                target_sizes=target_sizes,
            )[0]

            boxes = processed["boxes"]
            scores = processed["scores"]

            if len(scores) == 0:
                logger.debug(f"No detection for target '{target}' via GroundingDINO")
                continue

            # Pick highest-confidence box
            best_idx = scores.argmax().item()
            best_score = float(scores[best_idx].item())
            best_box = boxes[best_idx].tolist()  # [x1, y1, x2, y2] absolute pixels

            bbox = BoundingBox(
                x1=best_box[0] / w,
                y1=best_box[1] / h,
                x2=best_box[2] / w,
                y2=best_box[3] / h,
                confidence=best_score,
                label=prompt_text,
            )
            results[target] = bbox
            logger.debug(
                f"Detected '{target}' via DINO | conf={best_score:.3f} "
                f"| box=[{bbox.x1:.2f},{bbox.y1:.2f},{bbox.x2:.2f},{bbox.y2:.2f}]"
            )

        return results
