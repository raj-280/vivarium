"""
pipeline/measurers/clip_measurer.py

CLIP zero-shot measurer implementation.
Uses OpenAI CLIP to rank candidate labels and maps the best match
to a numeric level via the level_map / presence_map from config.

Zero hardcoded labels or levels — all read from config.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from dotmap import DotMap
from loguru import logger
from PIL import Image as PILImage

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


class CLIPMeasurer(BaseMeasurer):
    """Zero-shot level estimator using OpenAI CLIP."""

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._model = None
        self._preprocess = None
        self._device: str = "cpu"

    def load(self) -> None:
        """Load CLIP model and preprocessing pipeline."""
        try:
            import clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai-clip and torch are not installed. "
                "Run: pip install openai-clip torch"
            ) from exc

        target_cfg = getattr(self.config, self.target)
        clip_model_name: str = target_cfg.clip_model

        import torch

        self._device = self.config.detector.device  # reuse detector device setting
        logger.info(f"Loading CLIP model '{clip_model_name}' for target '{self.target}'")
        self._model, self._preprocess = clip.load(clip_model_name, device=self._device)
        self._model.eval()
        logger.info(f"CLIP model loaded for target '{self.target}'")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """
        Run CLIP zero-shot classification on the ROI.

        Returns:
            MeasurementResult with level derived from the ranked label index.
        """
        if self._model is None:
            raise RuntimeError(f"CLIPMeasurer for '{self.target}' not loaded — call .load() first")

        import clip  # type: ignore
        import torch  # type: ignore

        target_cfg = getattr(self.config, self.target)
        labels: List[str] = list(target_cfg.clip_labels)
        min_confidence = getattr(target_cfg, 'min_measurement_confidence', 0.55)

        # Convert BGR numpy to PIL RGB
        pil_image = PILImage.fromarray(roi[:, :, ::-1]).convert("RGB")
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)

        text_tokens = clip.tokenize(labels).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)
            probs = similarity.softmax(dim=-1)

        best_idx = int(probs.argmax().item())
        best_prob = float(probs[best_idx].item())
        best_label = labels[best_idx]

        # **OPTIMIZATION: Confidence gate — skip low-confidence predictions**
        if best_prob < min_confidence:
            logger.warning(
                f"CLIP [{self.target}] LOW CONFIDENCE: label='{best_label}' "
                f"conf={best_prob:.3f} < {min_confidence} — measurement rejected"
            )
            return MeasurementResult(
                level=None,
                confidence=best_prob,
                label=f"Low confidence ({best_prob:.2f}) — measurement uncertain",
                present=None,
            )

        level, present = self._resolve_level(target_cfg, best_idx)

        logger.debug(
            f"CLIP [{self.target}] → label='{best_label}' idx={best_idx} "
            f"conf={best_prob:.3f} level={level}"
        )

        return MeasurementResult(
            level=level,
            confidence=best_prob,
            label=best_label,
            present=present,
        )

    def _resolve_level(
        self, target_cfg: Any, best_idx: int
    ) -> tuple[float, bool | None]:
        """Map label index to numeric level and optional presence flag."""
        present: bool | None = None
        level: float = 0.0

        # Mouse uses presence_map, others use level_map
        if hasattr(target_cfg, "presence_map") and target_cfg.presence_map:
            presence_map = {int(k): v for k, v in target_cfg.presence_map.items()}
            present = presence_map.get(best_idx, False)
            level = 100.0 if present else 0.0
        elif hasattr(target_cfg, "level_map") and target_cfg.level_map:
            level_map = {int(k): float(v) for k, v in target_cfg.level_map.items()}
            level = level_map.get(best_idx, 0.0)

        return level, present
