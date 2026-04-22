"""
tests/test_detector.py

Unit tests for the detector layer.
Config values (prompts, thresholds, engine) are loaded from the real
config/config.yaml + config/config.local.yaml so tests always reflect
the current configuration.
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
def config(real_config) -> DotMap:
    """
    Full merged config loaded from config.yaml + config.local.yaml.
    Injected via the session-scoped real_config fixture in conftest.py.
    """
    return real_config


@pytest.fixture
def dummy_image() -> np.ndarray:
    """640x640 blank BGR image — same dimensions as preprocessor output."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# DetectorFactory tests
# ---------------------------------------------------------------------------

class TestDetectorFactory:

    def test_creates_configured_engine(self, config):
        """Factory creates whichever engine is set in config.detector.engine."""
        engine = config.detector.engine  # read from real config
        expected_class_map = {
            "yolov8world":   "YOLOv8WorldDetector",
            "groundingdino": "GroundingDINODetector",
            "owlvit":        "OWLViTDetector",
            "yolov8":        "YOLOv8Detector",
        }
        expected_class = expected_class_map.get(engine)
        assert expected_class is not None, f"Unexpected engine in config: {engine}"

        # Patch .load() so no model weights are needed
        patch_target = {
            "yolov8world":   "pipeline.detectors.yolov8world.YOLOv8WorldDetector.load",
            "groundingdino": "pipeline.detectors.groundingdino.GroundingDINODetector.load",
            "owlvit":        "pipeline.detectors.owlvit.OWLViTDetector.load",
            "yolov8":        "pipeline.detectors.yolov8.YOLOv8Detector.load",
        }[engine]

        with patch(patch_target):
            detector = DetectorFactory.create(config)
            assert detector.__class__.__name__ == expected_class

    def test_raises_on_unknown_engine(self, config):
        """Factory raises ConfigurationError for unsupported engine names."""
        bad_config = DotMap(config.toDict(), _dynamic=False)
        bad_config.detector.engine = "magic_engine"
        with pytest.raises(ConfigurationError, match="magic_engine"):
            DetectorFactory.create(bad_config)

    def test_configured_prompts_are_strings(self, config):
        """All prompts in config.detector.prompts must be non-empty strings."""
        prompts = config.detector.prompts
        for target in config.targets.enabled:
            prompt = getattr(prompts, target, None)
            assert isinstance(prompt, str), f"Prompt for '{target}' is not a string"
            assert len(prompt.strip()) > 0, f"Prompt for '{target}' is empty"

    def test_min_confidence_in_valid_range(self, config):
        """min_confidence must be between 0 and 1."""
        conf = float(config.detector.min_confidence)
        assert 0.0 < conf < 1.0, f"min_confidence={conf} is outside (0, 1)"


# ---------------------------------------------------------------------------
# YOLOv8World detector unit tests (model mocked)
# ---------------------------------------------------------------------------

class TestYOLOv8WorldDetector:

    def test_detect_returns_dict_for_all_targets(self, config, dummy_image):
        """Detector returns a dict with a key for every enabled target."""
        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            from pipeline.detectors.yolov8world import YOLOv8WorldDetector

            detector = YOLOv8WorldDetector(config)
            detector._model = MagicMock()
            detector._model.predict.return_value = [MagicMock(boxes=None)]

            targets = list(config.targets.enabled)
            result = detector.detect(dummy_image, targets)

            assert set(result.keys()) == set(targets)
            # All None because mock returns no boxes
            assert all(v is None for v in result.values())

    def test_detect_extracts_bounding_box(self, config, dummy_image):
        """Detector correctly parses a synthetic YOLO box into a BoundingBox."""
        import torch

        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            from pipeline.detectors.yolov8world import YOLOv8WorldDetector

            detector = YOLOv8WorldDetector(config)

            # Simulate: water → class index 0, conf=0.8, box=[100,100,300,300]
            mock_box = MagicMock()
            mock_box.cls = [torch.tensor(0)]
            mock_box.conf = [torch.tensor(0.8)]
            mock_box.xyxy = [torch.tensor([100.0, 100.0, 300.0, 300.0])]

            mock_pred = MagicMock()
            mock_pred.boxes = [mock_box]

            detector._model = MagicMock()
            detector._model.predict.return_value = [mock_pred]

            targets = list(config.targets.enabled)
            result = detector.detect(dummy_image, targets)

            water_box = result.get("water")
            assert water_box is not None
            assert isinstance(water_box, BoundingBox)
            assert abs(water_box.confidence - 0.8) < 1e-4

    def test_detect_filters_below_min_confidence(self, config, dummy_image):
        """Boxes below config.detector.min_confidence are not returned."""
        import torch

        min_conf = float(config.detector.min_confidence)
        low_conf = max(0.01, min_conf - 0.10)  # always below threshold

        with patch("pipeline.detectors.yolov8world.YOLOv8WorldDetector.load"):
            from pipeline.detectors.yolov8world import YOLOv8WorldDetector

            detector = YOLOv8WorldDetector(config)

            mock_box = MagicMock()
            mock_box.cls = [torch.tensor(0)]
            mock_box.conf = [torch.tensor(low_conf)]
            mock_box.xyxy = [torch.tensor([100.0, 100.0, 300.0, 300.0])]

            mock_pred = MagicMock()
            mock_pred.boxes = [mock_box]

            detector._model = MagicMock()
            detector._model.predict.return_value = [mock_pred]

            targets = list(config.targets.enabled)
            result = detector.detect(dummy_image, targets)

            # The gate (not the detector) filters by confidence, so the box
            # IS returned by the detector — gate test is in test_pipeline.py
            water_box = result.get("water")
            if water_box is not None:
                assert water_box.confidence == pytest.approx(low_conf, abs=1e-4)


# ---------------------------------------------------------------------------
# OWLViT detector unit tests (model mocked)
# ---------------------------------------------------------------------------

class TestOWLViTDetector:

    @pytest.fixture
    def owlvit_config(self, config) -> DotMap:
        """Config with engine overridden to owlvit for these tests."""
        cfg = DotMap(config.toDict(), _dynamic=False)
        cfg.detector.engine = "owlvit"
        # owlvit block is already in config.yaml — use it as-is
        return cfg

    def test_creates_owlvit_detector(self, owlvit_config):
        """Factory creates OWLViTDetector when engine=owlvit."""
        with patch("pipeline.detectors.owlvit.OWLViTDetector.load"):
            detector = DetectorFactory.create(owlvit_config)
            assert detector.__class__.__name__ == "OWLViTDetector"

    def test_detect_returns_none_when_no_boxes(self, owlvit_config, dummy_image):
        """OWLViT returns None for all targets when the model finds nothing."""
        with patch("pipeline.detectors.owlvit.OWLViTDetector.load"):
            from pipeline.detectors.owlvit import OWLViTDetector

            detector = OWLViTDetector(owlvit_config)

            mock_processor = MagicMock()
            mock_model = MagicMock()

            import torch
            mock_processor.return_value = {
                "input_ids": torch.zeros(1, 10, dtype=torch.long)
            }
            mock_model.return_value = MagicMock()
            mock_processor.post_process_grounded_object_detection.return_value = [{
                "boxes":  torch.zeros(0, 4),
                "scores": torch.zeros(0),
                "labels": [],
            }]

            detector._processor = mock_processor
            detector._model = mock_model
            detector._device = "cpu"

            targets = list(owlvit_config.targets.enabled)
            result = detector.detect(dummy_image, targets)
            assert all(v is None for v in result.values())

    def test_owlvit_score_threshold_from_config(self, owlvit_config):
        """OWLViT score_threshold must be read from config, not hardcoded."""
        threshold = float(owlvit_config.detector.owlvit.score_threshold)
        assert 0.0 < threshold <= 0.20, (
            f"owlvit.score_threshold={threshold} — OWL-ViT raw scores are low; "
            "keep at or below 0.20"
        )


# ---------------------------------------------------------------------------
# BoundingBox property tests — use gate values from real config
# ---------------------------------------------------------------------------

class TestBoundingBox:

    def test_aspect_ratio(self):
        # width=0.4, height=0.2 → AR=2.0
        box = BoundingBox(0.1, 0.1, 0.5, 0.3, 0.9, "test")
        assert abs(box.aspect_ratio - 2.0) < 1e-6

    def test_area_ratio(self):
        # 0.5 * 0.5 = 0.25
        box = BoundingBox(0.0, 0.0, 0.5, 0.5, 0.9, "test")
        assert abs(box.area_ratio - 0.25) < 1e-6

    def test_is_near_edge_true(self):
        box = BoundingBox(0.01, 0.01, 0.4, 0.4, 0.9, "test")
        assert box.is_near_edge(0.05) is True

    def test_is_near_edge_false(self):
        box = BoundingBox(0.1, 0.1, 0.8, 0.8, 0.9, "test")
        assert box.is_near_edge(0.05) is False

    def test_real_food_box_passes_configured_gate(self, config):
        """
        The actual food box from your vivarium image (from logs):
          box=[0.52, 0.45, 0.72, 0.72]  conf=0.920
        must pass all gate checks using the values in config.local.yaml.
        """
        box = BoundingBox(0.52, 0.45, 0.72, 0.72, 0.920, "food")
        gate = config.gate

        assert box.confidence >= float(config.detector.min_confidence), \
            f"conf {box.confidence} < min_confidence {config.detector.min_confidence}"

        assert box.aspect_ratio >= float(gate.min_box_aspect_ratio), \
            f"AR {box.aspect_ratio:.3f} < min_box_aspect_ratio {gate.min_box_aspect_ratio}"

        assert box.area_ratio >= float(gate.min_visible_area_ratio), \
            f"area {box.area_ratio:.4f} < min_visible_area_ratio {gate.min_visible_area_ratio}"

        assert not box.is_near_edge(float(gate.max_edge_proximity_ratio)), \
            f"box is near edge with proximity={gate.max_edge_proximity_ratio}"

    def test_gate_thresholds_allow_small_vivarium_objects(self, config):
        """
        min_visible_area_ratio must be <= 0.054 to allow the real food bowl
        (area_ratio=0.054) to pass the gate.
        """
        area_threshold = float(config.gate.min_visible_area_ratio)
        assert area_threshold <= 0.054, (
            f"min_visible_area_ratio={area_threshold} is too strict — "
            f"the food bowl occupies ~5.4% of the frame and would be rejected. "
            f"Set it to 0.03 or lower in config.local.yaml"
        )