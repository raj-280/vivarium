"""
tests/test_pipeline.py

Integration-level tests for the pipeline orchestrator.
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

def _make_config() -> DotMap:
    return DotMap(
        {
            "app": {"name": "vivarium-monitor", "env": "local", "log_level": "DEBUG"},
            "targets": {"enabled": ["water", "food", "mouse"]},
            "input": {
                "max_image_size_mb": 10,
                "allowed_formats": ["jpg", "jpeg", "png", "webp"],
            },
            "preprocessor": {
                "resize_to": [640, 640],
                "blur_threshold": 80,
                "normalize": True,
                "save_preprocessed": False,
            },
            "detector": {
                "engine": "yolov8world",
                "model_path": "./models/weights/yolov8x-world.pt",
                "device": "cpu",
                "min_confidence": 0.45,
                "prompts": {
                    "water": "water bottle",
                    "food": "food pile",
                    "mouse": "small mouse",
                },
            },
            "gate": {
                "enabled": True,
                "min_box_aspect_ratio": 1.4,
                "min_visible_area_ratio": 0.30,
                "max_edge_proximity_ratio": 0.05,
            },
            "water": {
                "engine": "clip",
                "clip_model": "ViT-B/32",
                "clip_labels": ["full", "half", "empty"],
                "level_map": {0: 100, 1: 50, 2: 0},
                "opencv": {"edge_method": "canny", "canny_threshold1": 50, "canny_threshold2": 150},
                "model_path": None,
            },
            "food": {
                "engine": "opencv",
                "clip_model": "ViT-B/32",
                "clip_labels": ["full", "half", "empty"],
                "level_map": {0: 100, 1: 50, 2: 0},
                "opencv": {"edge_method": "canny", "canny_threshold1": 50, "canny_threshold2": 150},
                "model_path": None,
            },
            "mouse": {
                "engine": "clip",
                "clip_model": "ViT-B/32",
                "clip_labels": ["mouse visible", "no mouse"],
                "presence_map": {0: True, 1: False},
                "level_map": {},
                "model_path": None,
            },
            "thresholds": {
                "water_low_pct": 25,
                "food_low_pct": 20,
                "mouse_missing_minutes": 60,
                "confidence_min": 0.60,
            },
            "notifiers": {
                "enabled": ["telegram"],
                "cooldown_minutes": 30,
                "templates": {
                    "water_low": "Water at {value}%",
                    "food_low": "Food at {value}%",
                    "mouse_missing": "Mouse missing for {minutes} min",
                    "image_rejected": "Rejected: {reason}",
                },
                "telegram": {
                    "bot_token": "FAKE_TOKEN",
                    "chat_id": "123456",
                    "parse_mode": "Markdown",
                },
                "email": {"smtp_host": "", "smtp_port": 587, "use_tls": True, "from": "", "to": "", "subject_prefix": ""},
                "webhook": {"url": None, "secret": None, "method": "POST", "timeout_seconds": 5},
            },
            "storage": {
                "engine": "postgres",
                "image_store": "local",
                "local_image_path": "./data/images",
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "db": "vivarium",
                    "user": "user",
                    "password": "pass",
                    "schema": "public",
                    "pool_size": 5,
                    "pool_timeout_seconds": 30,
                },
                "s3": {"bucket": None, "region": None, "prefix": ""},
                "gcs": {"bucket": None, "prefix": ""},
            },
        },
        _dynamic=False,
    )


def _make_valid_jpeg() -> bytes:
    """Encode a synthetic 640x640 image as JPEG bytes."""
    import cv2

    img = np.ones((640, 640, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestImageValidator:
    def test_accepts_valid_jpeg(self):
        from pipeline.preprocessor.image_validator import ImageValidator
        config = _make_config()
        validator = ImageValidator(config)
        img_bytes = _make_valid_jpeg()
        result = validator.validate(img_bytes)
        assert result is not None
        assert result.shape == (640, 640, 3)

    def test_rejects_oversized_file(self):
        from pipeline.preprocessor.image_validator import ImageValidator, ImageValidationError
        config = _make_config()
        config.input.max_image_size_mb = 0.001  # tiny limit
        validator = ImageValidator(config)
        with pytest.raises(ImageValidationError, match="exceeds limit"):
            validator.validate(_make_valid_jpeg())

    def test_rejects_blurry_image(self):
        from pipeline.preprocessor.image_validator import ImageValidator, ImageValidationError
        import cv2
        config = _make_config()
        config.preprocessor.blur_threshold = 9999  # impossibly high threshold
        validator = ImageValidator(config)
        img_bytes = _make_valid_jpeg()
        with pytest.raises(ImageValidationError, match="blurry"):
            validator.validate(img_bytes)


class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_run_returns_pipeline_result(self):
        """Full pipeline run with all components mocked."""
        config = _make_config()

        with (
            patch("pipeline.detectors.factory.DetectorFactory.create") as mock_det_factory,
            patch("pipeline.measurers.factory.MeasurerFactory.create") as mock_meas_factory,
            patch("pipeline.notifiers.factory.NotifierFactory.create_all") as mock_notif_factory,
            patch("pipeline.storage.factory.StorageFactory.create_db") as mock_db_factory,
            patch("pipeline.storage.factory.StorageFactory.create_image_store") as mock_img_factory,
        ):
            # Mock detector
            mock_detector = MagicMock()
            mock_detector.load = MagicMock()
            mock_detector.detect.return_value = {
                "water": BoundingBox(0.1, 0.1, 0.8, 0.9, 0.85, "water bottle"),
                "food": BoundingBox(0.1, 0.1, 0.8, 0.9, 0.75, "food pile"),
                "mouse": BoundingBox(0.1, 0.1, 0.8, 0.9, 0.80, "mouse"),
            }
            mock_det_factory.return_value = mock_detector

            # Mock measurer
            mock_measurer = MagicMock()
            mock_measurer.load = MagicMock()
            mock_measurer.measure.return_value = MeasurementResult(
                level=75.0, confidence=0.85, label="three quarters full", present=True
            )
            mock_meas_factory.return_value = mock_measurer

            # Mock notifiers
            mock_notif_factory.return_value = []

            # Mock storage
            mock_storage = MagicMock()
            mock_storage.connect = AsyncMock()
            mock_storage.disconnect = AsyncMock()
            mock_storage.save_result = AsyncMock(return_value="test-uuid-1234")
            mock_storage.update_image_path = AsyncMock()
            mock_storage.get_last_alert_at = AsyncMock(return_value=None)
            mock_storage.upsert_cooldown = AsyncMock()
            mock_storage.save_alert = AsyncMock()
            mock_db_factory.return_value = mock_storage

            # Mock image store
            mock_img_store = MagicMock()
            mock_img_store.save = AsyncMock(return_value="./data/images/test.jpg")
            mock_img_factory.return_value = mock_img_store

            from pipeline.orchestrator import PipelineOrchestrator

            orch = PipelineOrchestrator(config)
            await orch.startup()

            img_bytes = _make_valid_jpeg()
            result = await orch.run(img_bytes, "test.jpg")

            await orch.shutdown()

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.result_id == "test-uuid-1234"

    @pytest.mark.asyncio
    async def test_run_rejects_blurry_image(self):
        """Pipeline gracefully rejects a blurry image."""
        config = _make_config()
        config.preprocessor.blur_threshold = 9999

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
            mock_storage = MagicMock()
            mock_storage.connect = AsyncMock()
            mock_storage.disconnect = AsyncMock()
            mock_storage.get_last_alert_at = AsyncMock(return_value=None)
            mock_storage.upsert_cooldown = AsyncMock()
            mock_storage.save_alert = AsyncMock()
            mock_db_factory.return_value = mock_storage
            mock_img_factory.return_value = MagicMock(save=AsyncMock())

            from pipeline.orchestrator import PipelineOrchestrator

            orch = PipelineOrchestrator(config)
            await orch.startup()

            img_bytes = _make_valid_jpeg()
            result = await orch.run(img_bytes, "blurry.jpg")
            await orch.shutdown()

        assert result.success is False
        assert result.rejection_reason is not None
        assert "blurry" in result.rejection_reason.lower()
