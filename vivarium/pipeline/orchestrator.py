"""
pipeline/orchestrator.py

The pipeline orchestrator. Reads pipeline.stages from config and runs each
stage in sequence. All steps are defined in the FLOW doc-comment below.

Step 1  — Image validation (format, size, blur)
Step 2  — Preprocessing (resize + normalize)
Step 3  — Detector (factory → YOLOv8World | GroundingDINO)
Step 4  — Confidence + Angle Gate (per bounding box checks)
Step 5  — Measurers (one per enabled target, each factory-driven)
Step 6  — Result Aggregator (builds PipelineResult, saves to DB)
Step 7  — Threshold Engine (check thresholds, fire notifiers)
Step 8  — Storage (save image, update DB record)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from dotmap import DotMap
from loguru import logger

from core.result import BoundingBox, MeasurementResult, PipelineResult
from pipeline.detectors.base import BaseDetector
from pipeline.detectors.factory import DetectorFactory
from pipeline.measurers.base import BaseMeasurer
from pipeline.measurers.factory import MeasurerFactory
from pipeline.notifiers.base import BaseNotifier
from pipeline.notifiers.factory import NotifierFactory
from pipeline.preprocessor.image_validator import ImageValidationError, ImageValidator
from pipeline.preprocessor.resizer import ImageResizer
from pipeline.storage.base import BaseImageStore, BaseStorage
from pipeline.storage.factory import StorageFactory
from pipeline.threshold.engine import ThresholdEngine


class PipelineOrchestrator:
    """
    Central coordinator for the vivarium monitoring pipeline.

    Usage:
        orch = PipelineOrchestrator(config)
        await orch.startup()
        result = await orch.run(image_bytes, filename)
        await orch.shutdown()
    """

    def __init__(self, config: DotMap) -> None:
        self.config = config

        # Targets to measure
        self._targets: List[str] = list(config.targets.enabled)

        # Components (initialised in startup)
        self._validator: ImageValidator = ImageValidator(config)
        self._resizer: ImageResizer = ImageResizer(config)
        self._detector: BaseDetector = DetectorFactory.create(config)
        self._measurers: Dict[str, BaseMeasurer] = {}
        self._notifiers: List[BaseNotifier] = []
        self._storage: BaseStorage = StorageFactory.create_db(config)
        self._image_store: BaseImageStore = StorageFactory.create_image_store(config)
        self._threshold_engine: Optional[ThresholdEngine] = None

    async def startup(self) -> None:
        """Load all models and establish DB connection."""
        logger.info("Pipeline startup — loading models and connecting to DB")

        # Load detector
        self._detector.load()

        # Load measurers for each enabled target
        for target in self._targets:
            measurer = MeasurerFactory.create(self.config, target)
            measurer.load()
            self._measurers[target] = measurer

        # Create notifiers
        self._notifiers = NotifierFactory.create_all(self.config)

        # Connect to DB
        await self._storage.connect()

        # Build threshold engine
        self._threshold_engine = ThresholdEngine(
            self.config, self._storage, self._notifiers
        )

        logger.info(
            f"Pipeline ready | targets={self._targets} "
            f"| detector={self.config.detector.engine} "
            f"| notifiers={[n.__class__.__name__ for n in self._notifiers]}"
        )

    async def shutdown(self) -> None:
        """Disconnect from DB and release resources."""
        await self._storage.disconnect()
        logger.info("Pipeline shutdown complete")

    async def run(self, image_bytes: bytes, filename: str) -> PipelineResult:
        """
        Execute the full monitoring pipeline for one image.

        Args:
            image_bytes: Raw image bytes from the API upload.
            filename:    Original filename (used for image store key).

        Returns:
            PipelineResult dataclass with all measurements.
        """
        logger.info(f"Pipeline run started | filename={filename} | size={len(image_bytes)} bytes")

        # --- Step 1 + 2: Validate and preprocess ---
        try:
            raw_image = self._validator.validate(image_bytes)
        except ImageValidationError as exc:
            logger.warning(f"Image rejected: {exc.reason}")
            result = PipelineResult(
                success=False,
                rejection_reason=exc.reason,
                uncertain_targets=list(self._targets),
            )
            # Fire image_rejected alert
            if self._threshold_engine:
                await self._threshold_engine.fire_image_rejected(exc.reason)
            return result

        preprocessed = self._resizer.resize(raw_image)
        h, w = preprocessed.shape[:2]

        # --- Step 3: Detection ---
        detections: Dict[str, Optional[BoundingBox]] = self._detector.detect(
            preprocessed, self._targets
        )

        # --- Step 4: Gate ---
        gated: Dict[str, Optional[BoundingBox]] = {}
        uncertain_targets: List[str] = []

        if self.config.gate.enabled:
            for target, bbox in detections.items():
                if bbox is None:
                    logger.debug(f"Gate [{target}]: no detection — marking uncertain")
                    uncertain_targets.append(target)
                    gated[target] = None
                    continue

                # Confidence check
                if bbox.confidence < float(self.config.detector.min_confidence):
                    logger.debug(
                        f"Gate [{target}]: conf {bbox.confidence:.3f} < "
                        f"{self.config.detector.min_confidence} — uncertain"
                    )
                    uncertain_targets.append(target)
                    gated[target] = None
                    continue

                # Aspect ratio check
                if bbox.aspect_ratio < float(self.config.gate.min_box_aspect_ratio):
                    logger.debug(
                        f"Gate [{target}]: aspect_ratio {bbox.aspect_ratio:.2f} < "
                        f"{self.config.gate.min_box_aspect_ratio} — uncertain"
                    )
                    uncertain_targets.append(target)
                    gated[target] = None
                    continue

                # Visible area check
                if bbox.area_ratio < float(self.config.gate.min_visible_area_ratio):
                    logger.debug(
                        f"Gate [{target}]: area_ratio {bbox.area_ratio:.4f} < "
                        f"{self.config.gate.min_visible_area_ratio} — uncertain"
                    )
                    uncertain_targets.append(target)
                    gated[target] = None
                    continue

                # Edge proximity check
                edge_prox = float(self.config.gate.max_edge_proximity_ratio)
                if bbox.is_near_edge(edge_prox):
                    logger.debug(f"Gate [{target}]: too close to image edge — uncertain")
                    uncertain_targets.append(target)
                    gated[target] = None
                    continue

                gated[target] = bbox
        else:
            gated = detections

        # --- Step 5: Measurers ---
        measurements: Dict[str, MeasurementResult] = {}
        raw_detections: dict = {}

        for target in self._targets:
            bbox = gated.get(target)
            raw_detections[target] = bbox.to_dict() if bbox else None

            if bbox is None:
                continue  # skip measurement for uncertain targets

            # Crop ROI from bounding box (convert normalised to pixel coords)
            x1 = int(bbox.x1 * w)
            y1 = int(bbox.y1 * h)
            x2 = int(bbox.x2 * w)
            y2 = int(bbox.y2 * h)
            roi = preprocessed[y1:y2, x1:x2]

            if roi.size == 0:
                logger.warning(f"Empty ROI for target '{target}' — skipping measurement")
                uncertain_targets.append(target)
                continue

            measurer = self._measurers.get(target)
            if measurer is None:
                logger.warning(f"No measurer loaded for target '{target}'")
                continue

            try:
                meas = measurer.measure(roi)
                measurements[target] = meas
                logger.info(
                    f"Measurement [{target}]: level={meas.level:.1f} "
                    f"conf={meas.confidence:.3f} label='{meas.label}'"
                )
            except Exception as exc:
                logger.error(f"Measurer error for target '{target}': {exc}")
                uncertain_targets.append(target)

        # --- Step 6: Aggregate result ---
        result = PipelineResult(
            water_pct=measurements["water"].level if "water" in measurements else None,
            food_pct=measurements["food"].level if "food" in measurements else None,
            mouse_present=measurements["mouse"].present if "mouse" in measurements else None,
            water_confidence=measurements["water"].confidence if "water" in measurements else None,
            food_confidence=measurements["food"].confidence if "food" in measurements else None,
            mouse_confidence=measurements["mouse"].confidence if "mouse" in measurements else None,
            uncertain_targets=list(set(uncertain_targets)),
            raw_detections=raw_detections,
            success=True,
        )

        # Persist to DB
        result_id = await self._storage.save_result(result)
        result.result_id = result_id

        # --- Step 7: Threshold engine ---
        if self._threshold_engine:
            await self._threshold_engine.evaluate(result)

        # --- Step 8: Image storage ---
        stored_path = await self._image_store.save(image_bytes, filename)
        result.image_path = stored_path

        # Update DB record with image path
        await self._storage.update_image_path(result_id, stored_path)

        logger.info(
            f"Pipeline run complete | result_id={result_id} "
            f"| water={result.water_pct} food={result.food_pct} "
            f"| mouse={result.mouse_present}"
        )
        return result
