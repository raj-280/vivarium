"""
pipeline/measurers/classifier.py

Custom trained classifier measurer implementation.
Loads a custom ONNX or PyTorch model and runs inference.
All model path and class labels are read from config.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import MeasurementResult
from pipeline.measurers.base import BaseMeasurer


class ClassifierMeasurer(BaseMeasurer):
    """Level estimator using a custom trained classifier model (ONNX or PyTorch)."""

    def __init__(self, config: DotMap, target: str) -> None:
        super().__init__(config, target)
        self._session = None      # ONNX runtime session
        self._torch_model = None  # PyTorch model
        self._backend: str = "unknown"

    def load(self) -> None:
        """Load custom model. Tries ONNX first, then PyTorch (.pt/.pth)."""
        target_cfg = getattr(self.config, self.target)
        model_path_str: str = target_cfg.model_path

        if not model_path_str or model_path_str in ("null", "None", ""):
            raise ValueError(
                f"classifier engine selected for '{self.target}' but "
                f"{self.target}.model_path is not set in config.yaml."
            )

        model_path = Path(model_path_str)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Custom classifier model not found at '{model_path}'. "
                f"Check config.yaml → {self.target}.model_path"
            )

        suffix = model_path.suffix.lower()

        if suffix == ".onnx":
            self._load_onnx(model_path)
        elif suffix in (".pt", ".pth"):
            self._load_torch(model_path)
        else:
            raise ValueError(
                f"Unsupported model format '{suffix}' for target '{self.target}'. "
                "Use .onnx or .pt/.pth"
            )

    def _load_onnx(self, path: Path) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime") from exc

        logger.info(f"Loading ONNX classifier from {path} for target '{self.target}'")
        self._session = ort.InferenceSession(str(path))
        self._backend = "onnx"
        logger.info("ONNX classifier loaded")

    def _load_torch(self, path: Path) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("torch not installed. Run: pip install torch") from exc

        device = self.config.detector.device
        logger.info(f"Loading PyTorch classifier from {path} for target '{self.target}'")
        self._torch_model = torch.load(str(path), map_location=device)
        self._torch_model.eval()
        self._backend = "torch"
        logger.info("PyTorch classifier loaded")

    def measure(self, roi: np.ndarray) -> MeasurementResult:
        """Run custom classifier inference on the ROI."""
        import cv2  # type: ignore

        target_cfg = getattr(self.config, self.target)
        labels: List[str] = list(target_cfg.clip_labels)  # reuse clip_labels field

        # Resize to 224x224 and normalise
        resized = cv2.resize(roi, (224, 224))
        img_rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        # Normalise with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        img_chw = img_norm.transpose(2, 0, 1)  # HWC → CHW
        batch = img_chw[np.newaxis, ...]  # add batch dim → 1xCxHxW

        probs: np.ndarray

        if self._backend == "onnx":
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: batch})
            logits = outputs[0][0]
            probs = self._softmax(logits)
        elif self._backend == "torch":
            import torch  # type: ignore

            device = self.config.detector.device
            tensor = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                logits = self._torch_model(tensor)
            probs = self._softmax(logits.cpu().numpy()[0])
        else:
            raise RuntimeError("Classifier model not loaded — call .load() first")

        best_idx = int(probs.argmax())
        best_prob = float(probs[best_idx])
        best_label = labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"

        # Resolve level from level_map or presence_map
        level: float = 0.0
        present: bool | None = None

        if hasattr(target_cfg, "presence_map") and target_cfg.presence_map:
            presence_map = {int(k): v for k, v in target_cfg.presence_map.items()}
            present = presence_map.get(best_idx, False)
            level = 100.0 if present else 0.0
        elif hasattr(target_cfg, "level_map") and target_cfg.level_map:
            level_map = {int(k): float(v) for k, v in target_cfg.level_map.items()}
            level = level_map.get(best_idx, 0.0)

        logger.debug(
            f"Classifier [{self.target}] → label='{best_label}' "
            f"idx={best_idx} conf={best_prob:.3f} level={level}"
        )

        return MeasurementResult(
            level=level,
            confidence=best_prob,
            label=best_label,
            present=present,
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()
