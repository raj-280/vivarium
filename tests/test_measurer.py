"""
tests/test_measurer.py

Unit tests for the measurer layer.
Config values (clip_labels, level_map, min_measurement_confidence, engine)
are loaded from the real config/config.yaml + config/config.local.yaml.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from dotmap import DotMap

from core.result import MeasurementResult
from pipeline.measurers.factory import ConfigurationError, MeasurerFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(real_config) -> DotMap:
    """
    Full merged config from config.yaml + config.local.yaml.
    All measurer tests use labels, level_maps, and thresholds from here.
    """
    return real_config


@pytest.fixture
def blank_roi() -> np.ndarray:
    """200x200 blank BGR image as a stand-in ROI."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def gradient_roi() -> np.ndarray:
    """200x200 horizontal gradient — produces non-trivial Canny edges."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        img[:, i] = int(i * 255 / 200)
    return img


# ---------------------------------------------------------------------------
# Config sanity checks — catch misconfiguration before running the pipeline
# ---------------------------------------------------------------------------

class TestMeasurerConfig:

    def test_food_clip_labels_match_level_map(self, config):
        """Number of food CLIP labels must equal number of level_map entries."""
        food = config.food
        n_labels = len(list(food.clip_labels))
        n_levels = len(dict(food.level_map))
        assert n_labels == n_levels, (
            f"food has {n_labels} clip_labels but {n_levels} level_map entries — "
            f"they must match or the wrong level will be returned"
        )

    def test_water_clip_labels_match_level_map(self, config):
        """Number of water CLIP labels must equal number of level_map entries."""
        water = config.water
        n_labels = len(list(water.clip_labels))
        n_levels = len(dict(water.level_map))
        assert n_labels == n_levels, (
            f"water has {n_labels} clip_labels but {n_levels} level_map entries"
        )

    def test_mouse_clip_labels_match_presence_map(self, config):
        """Number of mouse CLIP labels must equal number of presence_map entries."""
        mouse = config.mouse
        n_labels = len(list(mouse.clip_labels))
        n_presence = len(dict(mouse.presence_map))
        assert n_labels == n_presence, (
            f"mouse has {n_labels} clip_labels but {n_presence} presence_map entries"
        )

    def test_food_min_confidence_allows_real_measurement(self, config):
        """
        food.min_measurement_confidence must be low enough to accept
        real CLIP scores (~0.25) on top-down vivarium images.
        """
        threshold = float(config.food.min_measurement_confidence)
        assert threshold <= 0.25, (
            f"food.min_measurement_confidence={threshold} is too high — "
            f"CLIP scores ~0.25 on top-down bowl images. "
            f"Set to 0.20 in config.local.yaml"
        )

    def test_all_enabled_targets_have_engine_configured(self, config):
        """Every enabled target must have an engine key set."""
        for target in config.targets.enabled:
            target_cfg = getattr(config, target, None)
            assert target_cfg is not None, f"No config section for target '{target}'"
            engine = getattr(target_cfg, "engine", None)
            assert engine is not None, f"No engine set for target '{target}'"
            assert engine in ("clip", "opencv", "classifier"), \
                f"Unknown measurer engine '{engine}' for target '{target}'"


# ---------------------------------------------------------------------------
# MeasurerFactory tests
# ---------------------------------------------------------------------------

class TestMeasurerFactory:

    def test_creates_measurer_for_each_enabled_target(self, config):
        """Factory creates a measurer for every target in config.targets.enabled."""
        for target in config.targets.enabled:
            measurer = MeasurerFactory.create(config, target)
            assert measurer is not None
            assert measurer.target == target

    def test_raises_on_unknown_engine(self, config):
        """Factory raises ConfigurationError for an unsupported engine value."""
        bad_config = DotMap(config.toDict(), _dynamic=False)
        bad_config.water.engine = "telepathy"
        with pytest.raises(ConfigurationError, match="telepathy"):
            MeasurerFactory.create(bad_config, "water")

    def test_raises_on_missing_target_section(self, config):
        """Factory raises ConfigurationError for a target with no config section."""
        with pytest.raises(ConfigurationError):
            MeasurerFactory.create(config, "oxygen")

    def test_food_measurer_class(self, config):
        """food engine from config determines the measurer class."""
        engine = config.food.engine
        measurer = MeasurerFactory.create(config, "food")
        expected = {
            "clip":       "CLIPMeasurer",
            "opencv":     "OpenCVMeasurer",
            "classifier": "ClassifierMeasurer",
        }[engine]
        assert measurer.__class__.__name__ == expected

    def test_mouse_measurer_class(self, config):
        """mouse engine from config determines the measurer class."""
        engine = config.mouse.engine
        measurer = MeasurerFactory.create(config, "mouse")
        expected = {
            "clip":       "CLIPMeasurer",
            "classifier": "ClassifierMeasurer",
        }[engine]
        assert measurer.__class__.__name__ == expected


# ---------------------------------------------------------------------------
# OpenCV measurer tests (no model loading needed)
# ---------------------------------------------------------------------------

class TestOpenCVMeasurer:

    @pytest.fixture
    def opencv_config(self, config) -> DotMap:
        """Config with food.engine forced to opencv for these tests."""
        cfg = DotMap(config.toDict(), _dynamic=False)
        cfg.food.engine = "opencv"
        return cfg

    def test_measure_blank_returns_result(self, opencv_config, blank_roi):
        """Blank image has no edges — returns a valid MeasurementResult."""
        measurer = MeasurerFactory.create(opencv_config, "food")
        measurer.load()
        result = measurer.measure(blank_roi)
        assert isinstance(result, MeasurementResult)
        assert 0.0 <= result.level <= 100.0
        assert 0.0 <= result.confidence <= 1.0

    def test_measure_gradient_returns_result(self, opencv_config, gradient_roi):
        """Gradient image has many edges — returns a valid MeasurementResult."""
        measurer = MeasurerFactory.create(opencv_config, "food")
        measurer.load()
        result = measurer.measure(gradient_roi)
        assert isinstance(result, MeasurementResult)
        assert result.level >= 0.0

    def test_unsupported_edge_method_raises(self, opencv_config, blank_roi):
        """Unsupported edge_method in config raises ValueError."""
        opencv_config.food.opencv.edge_method = "laplacian"
        measurer = MeasurerFactory.create(opencv_config, "food")
        measurer.load()
        with pytest.raises(ValueError, match="laplacian"):
            measurer.measure(blank_roi)


# ---------------------------------------------------------------------------
# CLIP measurer tests (model mocked)
# ---------------------------------------------------------------------------

class TestCLIPMeasurer:

    @pytest.fixture
    def clip_config(self, config) -> DotMap:
        """Config with water.engine forced to clip for these tests."""
        cfg = DotMap(config.toDict(), _dynamic=False)
        cfg.water.engine = "clip"
        return cfg

    def test_measure_returns_correct_level(self, clip_config, blank_roi):
        """
        Mock CLIP to return the third label (index 2).
        Expected level comes from config.water.level_map[2].
        """
        import torch

        level_map = {int(k): float(v) for k, v in clip_config.water.level_map.items()}
        expected_level = level_map[2]
        n_labels = len(list(clip_config.water.clip_labels))

        measurer = MeasurerFactory.create(clip_config, "water")

        with patch("clip.load") as mock_load:
            mock_model = MagicMock()
            mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_load.return_value = (mock_model, mock_preprocess)

            def mock_encode(x):
                t = torch.zeros(1, 512)
                t[0, 0] = 1.0
                return t

            mock_model.encode_image.side_effect = mock_encode
            mock_model.encode_text.side_effect = mock_encode

            # Force probs to pick index 2
            mock_probs = torch.zeros(n_labels)
            mock_probs[2] = 1.0

            with patch("torch.Tensor.softmax", return_value=mock_probs):
                measurer.load()
                measurer._model = mock_model
                measurer._preprocess = mock_preprocess
                result = measurer.measure(blank_roi)

        assert isinstance(result, MeasurementResult)
        assert result.level == pytest.approx(expected_level)

    def test_measure_returns_none_level_when_confidence_too_low(self, clip_config, blank_roi):
        """
        When CLIP confidence < min_measurement_confidence, the measurer
        must return level=None rather than a wrong value.
        """
        import torch

        # Force min_confidence very high so it always fails
        cfg = DotMap(clip_config.toDict(), _dynamic=False)
        cfg.water.min_measurement_confidence = 0.99

        n_labels = len(list(cfg.water.clip_labels))
        measurer = MeasurerFactory.create(cfg, "water")

        with patch("clip.load") as mock_load:
            mock_model = MagicMock()
            mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_load.return_value = (mock_model, mock_preprocess)

            mock_model.encode_image.return_value = torch.zeros(1, 512)
            mock_model.encode_text.return_value = torch.zeros(1, 512)

            # Spread confidence evenly — max will be 1/n_labels < 0.99
            mock_probs = torch.ones(n_labels) / n_labels

            with patch("torch.Tensor.softmax", return_value=mock_probs):
                measurer.load()
                measurer._model = mock_model
                measurer._preprocess = mock_preprocess
                result = measurer.measure(blank_roi)

        assert result.level is None, (
            "CLIPMeasurer must return level=None when confidence < min_measurement_confidence"
        )
        assert result.confidence < 0.99

    def test_mouse_presence_map_used_correctly(self, config, blank_roi):
        """
        Mouse measurer must use presence_map from config, not level_map.
        Index 0 → present=True, Index 1 → present=False.
        """
        import torch

        presence_map = {int(k): v for k, v in config.mouse.presence_map.items()}
        n_labels = len(list(config.mouse.clip_labels))

        cfg = DotMap(config.toDict(), _dynamic=False)
        cfg.mouse.engine = "clip"
        measurer = MeasurerFactory.create(cfg, "mouse")

        with patch("clip.load") as mock_load:
            mock_model = MagicMock()
            mock_preprocess = MagicMock(return_value=torch.zeros(3, 224, 224))
            mock_load.return_value = (mock_model, mock_preprocess)

            mock_model.encode_image.return_value = torch.zeros(1, 512)
            mock_model.encode_text.return_value = torch.zeros(1, 512)

            # Pick index 0 → "mouse visible" → present=True
            mock_probs = torch.zeros(n_labels)
            mock_probs[0] = 1.0

            with patch("torch.Tensor.softmax", return_value=mock_probs):
                measurer.load()
                measurer._model = mock_model
                measurer._preprocess = mock_preprocess
                result = measurer.measure(blank_roi)

        expected_present = presence_map.get(0, True)
        assert result.present == expected_present, (
            f"Mouse presence_map[0] should be {expected_present} per config"
        )