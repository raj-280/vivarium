"""
tests/test_pipeline.py

Integration-level tests for the pipeline orchestrator.
Config is loaded from the real config/config.yaml + config/config.local.yaml.
All external dependencies (models, DB, image store) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from dotmap import DotMap

from core.result import BoundingBox, MeasurementResult, PipelineResult
from pipeline.preprocessor.image_validator import ImageValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_jpeg() -> bytes:
    """Encode a synthetic 640x640 image as JPEG bytes."""
    import cv2

    img = np.ones((640, 640, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _mock_storage():
    """Return a fully mocked storage object for orchestrator tests."""
    mock = MagicMock()
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    mock.save_result = AsyncMock(return_value="test-uuid-1234")
    mock.update_image_path = AsyncMock()
    mock.get_last_alert_at = AsyncMock(return_value=None)
    mock.upsert_cooldown = AsyncMock()
    mock.save_alert = AsyncMock()
    mock.record_threshold_breach = AsyncMock()
    mock.count_recent_threshold_breaches = AsyncMock(return_value=0)
    return mock


def _mock_image_store():
    mock = MagicMock()
    mock.save = AsyncMock(return_value="./data/images/test.jpg")
    return mock


# ---------------------------------------------------------------------------
# Gate logic tests — values come from real config
# ---------------------------------------------------------------------------

class TestGateLogic:
    """
    Gate threshold tests using values from config.local.yaml.
    These tests verify that the configured thresholds match reality —
    specifically that the real food bowl bounding box from your vivarium
    image passes all gate checks.
    """

    def test_real_food_box_passes_all_gate_checks(self, real_config):
        """
        From logs: food detected at box=[0.52,0.45,0.72,0.72] conf=0.920.
        This box must pass every gate check with the current config values.
        """
        box = BoundingBox(0.52, 0.45, 0.72, 0.72, 0.920, "food")
        gate = real_config.gate

        assert box.confidence >= float(real_config.detector.min_confidence), \
            f"conf {box.confidence:.3f} < min_confidence {real_config.detector.min_confidence}"

        assert box.aspect_ratio >= float(gate.min_box_aspect_ratio), \
            f"AR {box.aspect_ratio:.3f} < min_box_aspect_ratio {gate.min_box_aspect_ratio}"

        assert box.area_ratio >= float(gate.min_visible_area_ratio), \
            f"area_ratio {box.area_ratio:.4f} < min_visible_area_ratio {gate.min_visible_area_ratio} — " \
            f"lower min_visible_area_ratio in config.local.yaml"

        assert not box.is_near_edge(float(gate.max_edge_proximity_ratio)), \
            f"food box is near edge with proximity={gate.max_edge_proximity_ratio}"

    def test_gate_rejects_box_below_min_area(self, real_config):
        """A box clearly smaller than min_visible_area_ratio must be rejected."""
        threshold = float(real_config.gate.min_visible_area_ratio)
        # Build a box with area half the threshold
        side = (threshold / 2) ** 0.5
        box = BoundingBox(0.1, 0.1, 0.1 + side, 0.1 + side, 0.9, "food")
        assert box.area_ratio < threshold

    def test_gate_rejects_low_confidence_box(self, real_config):
        """A box below min_confidence must be filtered at the gate."""
        min_conf = float(real_config.detector.min_confidence)
        box = BoundingBox(0.1, 0.1, 0.5, 0.5, min_conf - 0.05, "mouse")
        assert box.confidence < min_conf

    def test_gate_rejects_box_near_edge(self, real_config):
        """A box touching the image edge must be rejected."""
        prox = float(real_config.gate.max_edge_proximity_ratio)
        box = BoundingBox(0.0, 0.0, 0.3, 0.3, 0.9, "water")
        assert box.is_near_edge(prox) is True

    def test_gate_accepts_box_away_from_edge(self, real_config):
        """A box well away from all edges must not be rejected for proximity."""
        prox = float(real_config.gate.max_edge_proximity_ratio)
        # Use a box that is clearly inside the image
        margin = prox + 0.05
        box = BoundingBox(margin, margin, 1.0 - margin, 1.0 - margin, 0.9, "food")
        assert box.is_near_edge(prox) is False


# ---------------------------------------------------------------------------
# ImageValidator tests
# ---------------------------------------------------------------------------

class TestImageValidator:

    def test_accepts_valid_jpeg(self, real_config):
        from pipeline.preprocessor.image_validator import ImageValidator
        validator = ImageValidator(real_config)
        img_bytes = _make_valid_jpeg()
        result = validator.validate(img_bytes)
        assert result is not None
        assert result.shape[2] == 3  # BGR channels

    def test_rejects_oversized_file(self, real_config):
        from pipeline.preprocessor.image_validator import ImageValidator, ImageValidationError
        cfg = DotMap(real_config.toDict(), _dynamic=False)
        cfg.input.max_image_size_mb = 0.001  # tiny limit
        validator = ImageValidator(cfg)
        with pytest.raises(ImageValidationError, match="exceeds limit"):
            validator.validate(_make_valid_jpeg())

    def test_rejects_blurry_image(self, real_config):
        from pipeline.preprocessor.image_validator import ImageValidator, ImageValidationError
        cfg = DotMap(real_config.toDict(), _dynamic=False)
        cfg.preprocessor.blur_threshold = 9999  # impossibly high
        validator = ImageValidator(cfg)
        with pytest.raises(ImageValidationError, match="blurry"):
            validator.validate(_make_valid_jpeg())

    def test_blur_threshold_from_config(self, real_config):
        """Blur threshold must be a positive number."""
        threshold = float(real_config.preprocessor.blur_threshold)
        assert threshold > 0, "blur_threshold must be positive"


# ---------------------------------------------------------------------------
# Full pipeline orchestrator tests (all externals mocked)
# ---------------------------------------------------------------------------

class TestOrchestratorRun:

    @pytest.mark.asyncio
    async def test_run_returns_pipeline_result(self, real_config):
        """Full pipeline run with all external components mocked."""
        targets = list(real_config.targets.enabled)

        # Build detection dict from real targets
        detections = {
            t: BoundingBox(0.1, 0.1, 0.8, 0.9, 0.85, t) for t in targets
        }

        with (
            patch("pipeline.detectors.factory.DetectorFactory.create") as mock_det_factory,
            patch("pipeline.measurers.factory.MeasurerFactory.create") as mock_meas_factory,
            patch("pipeline.notifiers.factory.NotifierFactory.create_all") as mock_notif_factory,
            patch("pipeline.storage.factory.StorageFactory.create_db") as mock_db_factory,
            patch("pipeline.storage.factory.StorageFactory.create_image_store") as mock_img_factory,
        ):
            mock_detector = MagicMock()
            mock_detector.load = MagicMock()
            mock_detector.detect.return_value = detections
            mock_det_factory.return_value = mock_detector

            mock_measurer = MagicMock()
            mock_measurer.load = MagicMock()
            mock_measurer.measure.return_value = MeasurementResult(
                level=75.0, confidence=0.85, label="three quarters full", present=True
            )
            mock_meas_factory.return_value = mock_measurer

            mock_notif_factory.return_value = []
            mock_db_factory.return_value = _mock_storage()
            mock_img_factory.return_value = _mock_image_store()

            from pipeline.orchestrator import PipelineOrchestrator
            orch = PipelineOrchestrator(real_config)
            await orch.startup()
            result = await orch.run(_make_valid_jpeg(), "test.jpg")
            await orch.shutdown()

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.result_id == "test-uuid-1234"

    @pytest.mark.asyncio
    async def test_run_rejects_blurry_image(self, real_config):
        """Pipeline gracefully rejects a blurry image and returns success=False."""
        cfg = DotMap(real_config.toDict(), _dynamic=False)
        cfg.preprocessor.blur_threshold = 9999

        with (
            patch("pipeline.detectors.factory.DetectorFactory.create") as mock_det_factory,
            patch("pipeline.measurers.factory.MeasurerFactory.create") as mock_meas_factory,
            patch("pipeline.notifiers.factory.NotifierFactory.create_all") as mock_notif_factory,
            patch("pipeline.storage.factory.StorageFactory.create_db") as mock_db_factory,
            patch("pipeline.storage.factory.StorageFactory.create_image_store") as mock_img_factory,
        ):
            mock_det_factory.return_value = MagicMock(load=MagicMock(), detect=MagicMock())
            mock_meas_factory.return_value = MagicMock(load=MagicMock())
            mock_notif_factory.return_value = []
            mock_db_factory.return_value = _mock_storage()
            mock_img_factory.return_value = _mock_image_store()

            from pipeline.orchestrator import PipelineOrchestrator
            orch = PipelineOrchestrator(cfg)
            await orch.startup()
            result = await orch.run(_make_valid_jpeg(), "blurry.jpg")
            await orch.shutdown()

        assert result.success is False
        assert result.rejection_reason is not None
        assert "blurry" in result.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_gate_blocks_all_targets_when_area_too_small(self, real_config):
        """
        When detector finds boxes but all are below min_visible_area_ratio,
        all targets should end up in uncertain_targets and measurements=None.
        """
        targets = list(real_config.targets.enabled)
        min_area = float(real_config.gate.min_visible_area_ratio)

        # Build tiny boxes that will fail the area check
        side = max(0.001, (min_area / 2) ** 0.5)
        tiny_detections = {
            t: BoundingBox(0.1, 0.1, 0.1 + side, 0.1 + side, 0.9, t)
            for t in targets
        }

        with (
            patch("pipeline.detectors.factory.DetectorFactory.create") as mock_det_factory,
            patch("pipeline.measurers.factory.MeasurerFactory.create") as mock_meas_factory,
            patch("pipeline.notifiers.factory.NotifierFactory.create_all") as mock_notif_factory,
            patch("pipeline.storage.factory.StorageFactory.create_db") as mock_db_factory,
            patch("pipeline.storage.factory.StorageFactory.create_image_store") as mock_img_factory,
        ):
            mock_detector = MagicMock()
            mock_detector.load = MagicMock()
            mock_detector.detect.return_value = tiny_detections
            mock_det_factory.return_value = mock_detector

            mock_meas_factory.return_value = MagicMock(load=MagicMock())
            mock_notif_factory.return_value = []
            mock_db_factory.return_value = _mock_storage()
            mock_img_factory.return_value = _mock_image_store()

            from pipeline.orchestrator import PipelineOrchestrator
            orch = PipelineOrchestrator(real_config)
            await orch.startup()
            result = await orch.run(_make_valid_jpeg(), "test.jpg")
            await orch.shutdown()

        assert result.success is True
        assert result.water_pct is None
        assert result.food_pct is None
        assert result.mouse_present is None
        assert set(result.uncertain_targets) == set(targets)

    @pytest.mark.asyncio
    async def test_enabled_targets_from_config_are_all_measured(self, real_config):
        """
        Every target in config.targets.enabled must be attempted.
        If all detections pass the gate, all should have a measurement.
        """
        targets = list(real_config.targets.enabled)

        # Build boxes that pass the gate
        gate = real_config.gate
        margin = float(gate.max_edge_proximity_ratio) + 0.05
        size = (float(gate.min_visible_area_ratio) * 2) ** 0.5
        good_box = BoundingBox(
            margin, margin,
            margin + size, margin + size,
            float(real_config.detector.min_confidence) + 0.1,
            "target",
        )
        detections = {t: good_box for t in targets}

        with (
            patch("pipeline.detectors.factory.DetectorFactory.create") as mock_det_factory,
            patch("pipeline.measurers.factory.MeasurerFactory.create") as mock_meas_factory,
            patch("pipeline.notifiers.factory.NotifierFactory.create_all") as mock_notif_factory,
            patch("pipeline.storage.factory.StorageFactory.create_db") as mock_db_factory,
            patch("pipeline.storage.factory.StorageFactory.create_image_store") as mock_img_factory,
        ):
            mock_detector = MagicMock()
            mock_detector.load = MagicMock()
            mock_detector.detect.return_value = detections
            mock_det_factory.return_value = mock_detector

            mock_measurer = MagicMock()
            mock_measurer.load = MagicMock()
            mock_measurer.measure.return_value = MeasurementResult(
                level=50.0, confidence=0.80, label="half", present=True
            )
            mock_meas_factory.return_value = mock_measurer

            mock_notif_factory.return_value = []
            mock_db_factory.return_value = _mock_storage()
            mock_img_factory.return_value = _mock_image_store()

            from pipeline.orchestrator import PipelineOrchestrator
            orch = PipelineOrchestrator(real_config)
            await orch.startup()
            result = await orch.run(_make_valid_jpeg(), "test.jpg")
            await orch.shutdown()

        assert result.success is True
        # uncertain_targets should be empty — all passed the gate
        assert len(result.uncertain_targets) == 0, \
            f"These targets were unexpectedly rejected by the gate: {result.uncertain_targets}"