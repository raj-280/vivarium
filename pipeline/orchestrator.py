"""
pipeline/orchestrator.py

The pipeline orchestrator. Runs each stage in sequence.
Images uploaded via Swagger → pipeline runs → annotated image saved to output/ folder.

Phase 1 changes:
  Step 2 — UNDETERMINED confidence gate
  Step 4 — Duplicate frame detection via DeduplicatorFactory

Phase 2 changes:
  Step 2 — DB persistence (pipeline_results table)
  Step 2 — Alert evaluation (alert_log table, LOW / EMPTY states)

Phase 4/5 changes:
  Step 10 — runs are now called by TaskQueue workers (not directly from route)
  Step 12 — records latency + failure metrics via MetricsCollector

YOLOX update:
  - YOLOXMeasurer.set_last_label() is called before measure() so the measurer
    can decode the class ID from the BoundingBox label without a second model.
  - "bedding" target is handled and its fields populate PipelineResult.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pipeline.annotator.factory import AnnotatorFactory
from dotmap import DotMap
from loguru import logger

from core.alerting import AlertEvaluator
from core.database import get_session, is_db_ready
from core.frame_deduplicator import BaseDeduplicator, DeduplicatorFactory
from core.metrics import MetricsCollector
from core.pipeline_logger import PipelineEventLogger, PipelineRunContext
from core.repositories import insert_pipeline_result
from core.result import BoundingBox, MeasurementResult, PipelineResult
from core.mouse_stationary_tracker import MouseStationaryTracker
from core.level_labeler import label_water, label_food
from core.webhook import WebhookDispatcher
from pipeline.detectors.base import BaseDetector
from pipeline.detectors.factory import DetectorFactory
from pipeline.measurers.base import BaseMeasurer
from pipeline.measurers.factory import MeasurerFactory
from pipeline.preprocessor.image_validator import ImageValidationError, ImageValidator
from pipeline.preprocessor.resizer import ImageResizer

_UNDETERMINED = "UNDETERMINED"


def _get_min_confidence(config: DotMap, target: str) -> float:
    target_cfg = getattr(config.measurers, target, None)
    if target_cfg is not None:
        val = getattr(target_cfg, "min_confidence", None)
        if val is not None:
            return float(val)
    detector_conf = getattr(config.detector, "min_confidence", None)
    if detector_conf is not None:
        return float(detector_conf)
    return 0.0


def _is_yolox_measurer(measurer: BaseMeasurer) -> bool:
    """Return True if this measurer is a YOLOXMeasurer (has set_last_label)."""
    return hasattr(measurer, "set_last_label")


class PipelineOrchestrator:
    def __init__(self, config: DotMap) -> None:
        self.config = config
        self._targets: List[str] = list(config.targets.enabled)
        self._validator: ImageValidator = ImageValidator(config)
        self._resizer: ImageResizer = ImageResizer(config)
        self._detector: BaseDetector = DetectorFactory.create(config)
        self._measurers: Dict[str, BaseMeasurer] = {}
        self._stationary_tracker = MouseStationaryTracker(config)
        self._event_logger = PipelineEventLogger(config)
        self._webhook = WebhookDispatcher(config)
        self._deduplicator: BaseDeduplicator = DeduplicatorFactory.create(config)
        self._alert_evaluator = AlertEvaluator(config)
        self._metrics: Optional[MetricsCollector] = None
        self._min_confidence: Dict[str, float] = {
            t: _get_min_confidence(config, t) for t in self._targets
        }

        logger.info(
            f"PipelineOrchestrator initialised | targets={self._targets} "
            f"| detector={config.detector.engine} "
            f"| min_confidence={self._min_confidence}"
        )

    def set_metrics(self, metrics: MetricsCollector) -> None:
        self._metrics = metrics

    async def startup(self) -> None:
        logger.info("Pipeline startup — loading models")
        self._detector.load()
        for target in self._targets:
            measurer = MeasurerFactory.create(config=self.config, target=target)
            measurer.load()
            self._measurers[target] = measurer
        logger.info(
            f"Pipeline ready | targets={self._targets} "
            f"| detector={self.config.detector.engine}"
        )

    async def shutdown(self) -> None:
        logger.info("Pipeline shutdown complete")

    async def run(self, image_bytes: bytes, filename: str, cage_id: str) -> PipelineResult:
        _start = time.perf_counter()

        ctx = PipelineRunContext(
            cage_id=cage_id,
            filename=filename,
            image_size_bytes=len(image_bytes),
            image_received_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            f"Pipeline run started | cage={cage_id} | filename={filename} "
            f"| size={len(image_bytes)} bytes"
        )

        # ── Duplicate frame check ──────────────────────────────────────
        is_dup, dup_reason = self._deduplicator.check(cage_id, image_bytes)
        if is_dup:
            logger.info(f"Duplicate frame rejected | cage={cage_id} | {dup_reason}")
            result = PipelineResult(
                success=False,
                rejection_reason=f"duplicate_frame: {dup_reason}",
                uncertain_targets=list(self._targets),
            )
            ctx.duplicate_frame = True
            ctx.finish(result)
            self._event_logger.log(ctx)
            return result

        # ── Validate ──────────────────────────────────────────────────
        try:
            raw_image = self._validator.validate(image_bytes)
        except ImageValidationError as exc:
            logger.warning(f"Image rejected: {exc.reason}")
            result = PipelineResult(
                success=False,
                rejection_reason=exc.reason,
                uncertain_targets=list(self._targets),
            )
            ctx.finish(result)
            self._event_logger.log(ctx)
            return result

        # ── Preprocess ────────────────────────────────────────────────
        preprocessed = self._resizer.resize(raw_image)
        h, w = preprocessed.shape[:2]

        # ── Detect ────────────────────────────────────────────────────
        detections: Dict[str, Optional[BoundingBox]] = self._detector.detect(
            preprocessed, self._targets
        )

        # ── Measure ───────────────────────────────────────────────────
        measurements: Dict[str, MeasurementResult] = {}
        uncertain_targets: List[str] = []
        raw_detections: dict = {}

        for target in self._targets:
            bbox = detections.get(target)
            raw_detections[target] = bbox.to_dict() if bbox else None

            if bbox is None:
                logger.debug(f"No detection for '{target}' — skipping measurement")
                uncertain_targets.append(target)
                continue

            x1, y1 = int(bbox.x1 * w), int(bbox.y1 * h)
            x2, y2 = int(bbox.x2 * w), int(bbox.y2 * h)
            roi = preprocessed[y1:y2, x1:x2]

            if roi.size == 0:
                logger.warning(f"Empty ROI for '{target}' — skipping measurement")
                uncertain_targets.append(target)
                continue

            measurer = self._measurers.get(target)
            if measurer is None:
                logger.warning(f"No measurer loaded for '{target}'")
                continue

            # ── YOLOX: inject bbox label before measure() ──────────────
            # YOLOXMeasurer decodes level from the class ID in bbox.label.
            # All other measurers ignore set_last_label — it doesn't exist on them.
            if _is_yolox_measurer(measurer):
                area_pct = bbox.area_ratio   # width * height (already normalised 0–1)
                measurer.set_last_label(bbox.label, area_pct=area_pct, confidence=bbox.confidence)

            try:
                meas = measurer.measure(roi)
            except Exception as exc:
                logger.error(f"Measurer error for '{target}': {exc}")
                uncertain_targets.append(target)
                continue

            # ── Confidence gate ────────────────────────────────────────
            min_conf = self._min_confidence.get(target, 0.0)
            if min_conf > 0.0 and meas.confidence < min_conf:
                logger.warning(
                    f"[{target}] confidence {meas.confidence:.3f} < "
                    f"threshold {min_conf:.3f} → UNDETERMINED"
                )
                meas = MeasurementResult(
                    level=meas.level,
                    confidence=meas.confidence,
                    label=_UNDETERMINED,
                    present=meas.present,
                    bedding_condition=meas.bedding_condition,
                    bedding_area_pct=meas.bedding_area_pct,
                )
                uncertain_targets.append(target)
                measurements[target] = meas
                continue

            measurements[target] = meas
            logger.info(
                f"Measurement [{target}]: level={meas.level} "
                f"conf={meas.confidence:.3f} label='{meas.label}'"
            )

        # ── Annotate ──────────────────────────────────────────────────
        annotated_path = None
        try:
            annotator = AnnotatorFactory.create(self.config)
            annotated_path = annotator.draw(
                preprocessed, detections, measurements,
                result_id="pending",
                filename=filename,
            )
            logger.info(f"Annotated image saved → {annotated_path}")
        except Exception as exc:
            logger.warning(f"Annotator failed (non-fatal): {exc}", exc_info=True)

        # ── Mouse stationary check ─────────────────────────────────────
        mouse_bbox = detections.get("mouse")
        pre_state = self._stationary_tracker.get_state(cage_id)
        ctx.mouse_previous_bbox = pre_state.last_bbox if pre_state else None
        mouse_stationary = self._stationary_tracker.check(cage_id=cage_id, bbox=mouse_bbox)
        post_state = self._stationary_tracker.get_state(cage_id)
        if post_state is not None:
            ctx.mouse_consecutive_count = post_state.consecutive_count

        # ── Level labels ──────────────────────────────────────────────
        water_meas = measurements.get("water")
        food_meas  = measurements.get("food")

        water_pct = (
            water_meas.level
            if water_meas is not None and water_meas.label not in (_UNDETERMINED, "MISSING")
            else None
        )
        food_pct = (
            food_meas.level
            if food_meas is not None and food_meas.label not in (_UNDETERMINED,)
            else None
        )

        water_label = label_water(water_pct, self.config) if water_pct is not None else (
            water_meas.label if water_meas is not None else _UNDETERMINED
        )
        food_label = label_food(food_pct, self.config) if food_pct is not None else (
            food_meas.label if food_meas is not None else _UNDETERMINED
        )

        # ── Bedding fields ─────────────────────────────────────────────
        bedding_meas = measurements.get("bedding")
        bedding_condition  = bedding_meas.bedding_condition  if bedding_meas is not None else None
        bedding_confidence = bedding_meas.confidence         if bedding_meas is not None else None
        bedding_area_pct   = bedding_meas.bedding_area_pct   if bedding_meas is not None else None

        logger.info(
            f"Labels | water={water_label} ({water_pct}) "
            f"| food={food_label} ({food_pct}) "
            f"| bedding={bedding_condition}"
        )

        # ── Aggregate result ───────────────────────────────────────────
        result = PipelineResult(
            water_pct=water_pct,
            food_pct=food_pct,
            water_label=water_label,
            food_label=food_label,
            mouse_present=measurements["mouse"].present if "mouse" in measurements else None,
            mouse_stationary=mouse_stationary,
            water_confidence=water_meas.confidence if water_meas is not None else None,
            food_confidence=food_meas.confidence   if food_meas  is not None else None,
            mouse_confidence=measurements["mouse"].confidence if "mouse" in measurements else None,
            bedding_condition=bedding_condition,
            bedding_confidence=bedding_confidence,
            bedding_area_pct=bedding_area_pct,
            uncertain_targets=list(set(uncertain_targets)),
            raw_detections=raw_detections,
            image_path=annotated_path,
            success=True,
        )

        elapsed_ms = (time.perf_counter() - _start) * 1000
        logger.info(
            f"Pipeline complete | "
            f"water={result.water_pct} food={result.food_pct} "
            f"mouse={result.mouse_present} bedding={result.bedding_condition} "
            f"| elapsed={elapsed_ms:.0f}ms"
        )

        # ── Persist result + evaluate alerts ──────────────────────────
        fired_alerts = []
        if is_db_ready():
            try:
                async with get_session() as session:
                    result_id = await insert_pipeline_result(session, result, cage_id)
                    result.result_id = result_id
                    fired_alerts = await self._alert_evaluator.evaluate(
                        session, result, cage_id
                    )
                    if fired_alerts:
                        logger.info(
                            f"Alerts fired | cage={cage_id} | count={len(fired_alerts)} "
                            f"| targets={[a.target for a in fired_alerts]}"
                        )
            except Exception as exc:
                logger.error(f"DB persist/alert error (non-fatal): {exc}")

        # ── Structured event log ───────────────────────────────────────
        ctx.finish(result)

        # ── Webhook ───────────────────────────────────────────────────
        webhook_success = False
        if self._webhook.enabled:
            try:
                webhook_success = await self._webhook.dispatch(cage_id, result, fired_alerts)
                ctx.webhook_fired = webhook_success
            except Exception as exc:
                logger.error(f"Webhook dispatch error (non-fatal): {exc}")

        # ── Metrics ───────────────────────────────────────────────────
        if self._metrics is not None:
            elapsed_total_ms = (time.perf_counter() - _start) * 1000
            self._metrics.record_run(
                success=result.success,
                latency_ms=elapsed_total_ms,
            )
            if self._webhook.enabled:
                self._metrics.record_webhook(success=webhook_success)

        self._event_logger.log(ctx)

        return result
