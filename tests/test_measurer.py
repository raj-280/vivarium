"""
tests/test_measurer.py

Unit tests for the measurer layer.
Uses fixture images from tests/fixtures/ where possible.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dotmap import DotMap

from core.result import MeasurementResult
from pipeline.measurers.factory import ConfigurationError, MeasurerFactory


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> DotMap:
    return DotMap(
        {
            "detector": {"device": "cpu"},
            "water": {
                "engine": "clip",
                "clip_model": "ViT-B/32",
                "clip_labels": [
                    "water bottle completely full",
                    "water bottle three quarters full",
                    "water bottle half full",
                    "water bottle nearly empty",
                    "water bottle empty",
                ],
                "level_map": {0: 100, 1: 75, 2: 50, 3: 20, 4: 0},
                "opencv": {
                    "edge_method": "canny",
                    "canny_threshold1": 50,
                    "canny_threshold2": 150,
                },
                "model_path": None,
            },
            "food": {
                "engine": "opencv",
                "clip_model": "ViT-B/32",
                "clip_labels": [
                    "food tray completely full",
                    "food tray half full",
                    "food tray low on food",
                    "food tray empty",
                ],
                "level_map": {0: 100, 1: 50, 2: 15, 3: 0},
                "opencv": {
                    "edge_method": "canny",
                    "canny_threshold1": 50,
                    "canny_threshold2": 150,
                },
                "model_path": None,
            },
            "mouse": {
                "engine": "clip",
                "clip_model": "ViT-B/32",
                "clip_labels": [
                    "mouse visible in the cage",
                    "no mouse visible in the cage",
                ],
                "presence_map": {0: True, 1: False},
                "level_map": {},
                "model_path": None,
            },
        },
        _dynamic=False,
    )


@pytest.fixture
def blank_roi() -> np.ndarray:
    """200x200 blank BGR image as a stand-in ROI."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def gradient_roi() -> np.ndarray:
    """200x200 gradient image — produces non-trivial Canny edges."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        img[:, i] = int(i * 255 / 200)
    return img


# ---------------------------------------------------------------------------
# MeasurerFactory tests
# ---------------------------------------------------------------------------

class TestMeasurerFactory:
    def test_creates_clip_measurer(self, base_config):
        measurer = MeasurerFactory.create(base_config, "water")
        assert measurer.__class__.__name__ == "CLIPMeasurer"

    def test_creates_opencv_measurer(self, base_config):
        measurer = MeasurerFactory.create(base_config, "food")
        assert measurer.__class__.__name__ == "OpenCVMeasurer"

    def test_raises_on_unknown_engine(self, base_config):
        base_config.water.engine = "telepathy"
        with pytest.raises(ConfigurationError, match="telepathy"):
            MeasurerFactory.create(base_config, "water")

    def test_raises_on_missing_target_section(self, base_config):
        with pytest.raises(ConfigurationError, match="oxygen"):
            MeasurerFactory.create(base_config, "oxygen")


# ---------------------------------------------------------------------------
# OpenCV measurer tests (no model loading needed)
# ---------------------------------------------------------------------------

class TestOpenCVMeasurer:
    def test_measure_blank_returns_result(self, base_config, blank_roi):
        """Blank image has no edges — level should be 0 or very low."""
        measurer = MeasurerFactory.create(base_config, "food")
        measurer.load()  # safe — just imports cv2
        result = measurer.measure(blank_roi)
        assert isinstance(result, MeasurementResult)
        assert 0.0 <= result.level <= 100.0
        assert 0.0 <= result.confidence <= 1.0

    def test_measure_gradient_returns_result(self, base_config, gradient_roi):
        """Gradient image has many edges."""
        measurer = MeasurerFactory.create(base_config, "food")
        measurer.load()
        result = measurer.measure(gradient_roi)
        assert isinstance(result, MeasurementResult)
        assert result.level >= 0.0

    def test_unsupported_edge_method_raises(self, base_config, blank_roi):
        base_config.food.opencv.edge_method = "laplacian"
        measurer = MeasurerFactory.create(base_config, "food")
        measurer.load()
        with pytest.raises(ValueError, match="laplacian"):
            measurer.measure(blank_roi)


# ---------------------------------------------------------------------------
# CLIP measurer tests (model mocked)
# ---------------------------------------------------------------------------

class TestCLIPMeasurer:
    def test_measure_returns_correct_level(self, base_config, blank_roi):
        """Mock CLIP to return index 2 → level 50 for water."""
        import torch

        measurer = MeasurerFactory.create(base_config, "water")

        with patch("clip.load") as mock_load:
            mock_model = MagicMock()
            mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_load.return_value = (mock_model, mock_preprocess)

            # Make encode_image and encode_text return unit vectors
            def mock_encode(x):
                t = torch.zeros(1, 512)
                t[0, 0] = 1.0
                return t

            mock_model.encode_image.side_effect = mock_encode
            mock_model.encode_text.side_effect = mock_encode

            # Force probs to pick index 2 (half full → 50%)
            mock_probs = torch.zeros(5)
            mock_probs[2] = 1.0

            with patch("torch.Tensor.softmax", return_value=mock_probs):
                measurer.load()
                measurer._model = mock_model
                measurer._preprocess = mock_preprocess

                result = measurer.measure(blank_roi)

        assert isinstance(result, MeasurementResult)
