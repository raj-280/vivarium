"""
tests/test_detector.py

Unit tests for the detector layer.
Uses a stub detector to avoid loading real model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dotmap import DotMap

from core.result import BoundingBox
from pipeline.detectors.factory import ConfigurationError, DetectorFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> DotMap:
    return DotMap(
        {
            "detector": {
                "engine": "yolov8world",
                "model_path": "./models/weights/yolov8x-world.pt",
                "device": "cpu",
                "min_confidence": 0.45,
                "prompts": {
                    "water": "water bottle with liquid inside",
                    "food": "food pile in a feeding tray",
                    "mouse": "small white mouse on white bedding",
                },
            },
            "targets": {"enabled": ["water", "food", "mouse"]},
        },
        _dynamic=False,
    )


@pytest.fixture
def dummy_image() -> np.ndarray:
    """640x640 blank BGR image."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# DetectorFactory tests
# ---------------------------------------------------------------------------

class TestDetectorFactory:
    def test_creates_yolov8world(self, base_config):
        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            detector = DetectorFactory.create(base_config)
            assert detector.__class__.__name__ == "YOLOv8WorldDetector"

    def test_creates_groundingdino(self, base_config):
        base_config.detector.engine = "groundingdino"
        with patch("pipeline.detectors.groundingdino.GroundingDINODetector.load"):
            detector = DetectorFactory.create(base_config)
            assert detector.__class__.__name__ == "GroundingDINODetector"

    def test_raises_on_unknown_engine(self, base_config):
        base_config.detector.engine = "magic_engine"
        with pytest.raises(ConfigurationError, match="magic_engine"):
            DetectorFactory.create(base_config)


# ---------------------------------------------------------------------------
# YOLOv8World detector unit tests (model mocked)
# ---------------------------------------------------------------------------

class TestYOLOv8WorldDetector:
    def test_detect_returns_dict_for_all_targets(self, base_config, dummy_image):
        """Detector returns a dict keyed by target name."""
        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            from pipeline.detectors.yolov8world import YOLOv8WorldDetector

            detector = YOLOv8WorldDetector(base_config)
            detector._model = MagicMock()
            detector._model.predict.return_value = [MagicMock(boxes=None)]

            result = detector.detect(dummy_image, ["water", "food", "mouse"])
            assert set(result.keys()) == {"water", "food", "mouse"}
            # All None because mock returns no boxes
            assert all(v is None for v in result.values())

    def test_detect_extracts_bounding_box(self, base_config, dummy_image):
        """Detector correctly parses a synthetic YOLO box."""
        import torch

        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            from pipeline.detectors.yolov8world import YOLOv8WorldDetector

            detector = YOLOv8WorldDetector(base_config)

            # Build mock box: water → index 0, conf=0.8, box=[100,100,300,300]
            mock_box = MagicMock()
            mock_box.cls = [torch.tensor(0)]
            mock_box.conf = [torch.tensor(0.8)]
            mock_box.xyxy = [torch.tensor([100.0, 100.0, 300.0, 300.0])]

            mock_pred = MagicMock()
            mock_pred.boxes = [mock_box]

            detector._model = MagicMock()
            detector._model.predict.return_value = [mock_pred]

            result = detector.detect(dummy_image, ["water", "food", "mouse"])
            water_box = result.get("water")
            assert water_box is not None
            assert isinstance(water_box, BoundingBox)
            assert abs(water_box.confidence - 0.8) < 1e-4


# ---------------------------------------------------------------------------
# BoundingBox property tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_aspect_ratio(self):
        box = BoundingBox(0.1, 0.1, 0.5, 0.3, 0.9, "test")
        assert abs(box.aspect_ratio - 2.0) < 1e-6

    def test_area_ratio(self):
        box = BoundingBox(0.0, 0.0, 0.5, 0.5, 0.9, "test")
        assert abs(box.area_ratio - 0.25) < 1e-6

    def test_is_near_edge_true(self):
        box = BoundingBox(0.01, 0.01, 0.4, 0.4, 0.9, "test")
        assert box.is_near_edge(0.05) is True

    def test_is_near_edge_false(self):
        box = BoundingBox(0.1, 0.1, 0.8, 0.8, 0.9, "test")
        assert box.is_near_edge(0.05) is False
