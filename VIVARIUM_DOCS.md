# Vivarium Monitoring Pipeline — Full Project Documentation

> **What this project does:** Receives images of a small animal vivarium (water bottle, food bowl, mouse),
> runs a computer vision pipeline to measure water level, food level, and mouse presence/stationarity,
> persists results to a database, fires threshold alerts, sends webhook notifications, and returns
> everything via a FastAPI REST API.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Architecture Overview](#2-architecture-overview)
3. [Setup & Installation](#3-setup--installation)
4. [Configuration Reference](#4-configuration-reference)
5. [API Endpoints](#5-api-endpoints)
6. [Pipeline Stages (Orchestrator)](#6-pipeline-stages-orchestrator)
7. [Core Modules](#7-core-modules)
8. [Job Tracking & Task Queue](#8-job-tracking--task-queue)
9. [Database & Persistence](#9-database--persistence)
10. [Alerting System](#10-alerting-system)
11. [Webhooks](#11-webhooks)
12. [Metrics](#12-metrics)
13. [Detectors](#13-detectors)
14. [Measurers](#14-measurers)
15. [Annotator](#15-annotator)
16. [Frame Deduplication](#16-frame-deduplication)
17. [Mouse Stationary Tracker](#17-mouse-stationary-tracker)
18. [Docker Deployment](#18-docker-deployment)
19. [Adding a New Engine (Extensibility Guide)](#19-adding-a-new-engine-extensibility-guide)
20. [Known Limitations & Future Work](#20-known-limitations--future-work)

---

## 1. Project Structure

```
vivarium/
│
├── api/                        # FastAPI application layer
│   ├── main.py                 # App factory, lifespan (startup/shutdown), singletons
│   ├── dependencies.py         # FastAPI dependency injectors for shared singletons
│   ├── errors.py               # Global error handlers
│   └── routes/
│       ├── ingest.py           # POST /analyze  — file upload → async queue
│       ├── ingest_s3.py        # POST /analyze/s3 — S3 image → async queue
│       ├── jobs.py             # GET  /job/{request_id} — poll job status
│       ├── results.py          # GET  /results — query DB history
│       └── metrics.py          # GET  /metrics — live pipeline metrics
│
├── core/                       # Shared infrastructure (no pipeline logic)
│   ├── job_tracker.py          # In-memory PENDING/PROCESSING/DONE/FAILED state machine
│   ├── task_queue.py           # asyncio.Queue + N worker coroutines
│   ├── database.py             # SQLAlchemy async engine, session factory, ORM models
│   ├── repositories.py         # DB read/write helpers (pipeline_results, alert_log)
│   ├── alerting.py             # Threshold evaluation → alert_log rows
│   ├── metrics.py              # Thread-safe in-memory counters
│   ├── webhook.py              # Fire-and-forget async HTTP POST dispatcher
│   ├── frame_deduplicator.py   # MD5 + pHash duplicate frame detection
│   ├── mouse_stationary_tracker.py  # IoU-based consecutive-position tracker
│   ├── level_labeler.py        # Numeric % → human label (Critical/Low/OK/Full)
│   ├── result.py               # Shared dataclasses: BoundingBox, MeasurementResult, PipelineResult
│   ├── s3_fetcher.py           # boto3 wrapper — fetch image bytes from S3
│   ├── config_loader.py        # Loads + merges config.yaml → DotMap
│   ├── logger.py               # Loguru setup
│   └── pipeline_logger.py      # Structured per-run JSONL event log
│
├── pipeline/                   # All CV logic
│   ├── orchestrator.py         # Master coordinator — runs all stages in order
│   ├── preprocessor/
│   │   ├── image_validator.py  # Format, size, blur checks
│   │   └── resizer.py          # Resize + optional CLAHE brightness normalisation
│   ├── detectors/
│   │   ├── base.py             # BaseDetector ABC
│   │   ├── factory.py          # DetectorFactory — selects engine from config
│   │   ├── yolox_detector.py   # YOLOX 13-class detector (ACTIVE default)
│   │   ├── yolov8world.py      # YOLOv8-World open-vocabulary detector (switchable)
│   │   └── yolov8.py           # Standard YOLOv8 closed-vocabulary detector (switchable)
│   ├── measurers/
│   │   ├── base.py             # BaseMeasurer ABC
│   │   ├── factory.py          # MeasurerFactory — selects engine per target from config
│   │   ├── yolox_measurer.py   # Decodes level from YOLOX class ID — no model needed (ACTIVE default)
│   │   ├── opencv_water_measurer.py  # Water level via edge/contour detection
│   │   ├── opencv_food_measurer.py   # Food level via colour/region analysis
│   │   ├── unet_measurer.py          # UNet segmentation (water + food)
│   │   ├── pspnet_measurer.py        # PSPNet segmentation (requires water_pspnet.pt)
│   │   ├── classifier.py             # ONNX classifier model
│   │   └── detection_presence_measurer.py  # Mouse: present if detector fired
│   └── annotator/
│       ├── base.py             # BaseAnnotator ABC
│       ├── factory.py          # AnnotatorFactory
│       └── opencv_annotator.py # Draws bounding boxes + labels on output image
│
├── ml_models/                  # PyTorch model architecture definitions
│   ├── water_unet.py           # UNet architecture for water segmentation
│   ├── food_unet.py            # UNet architecture for food segmentation
│   └── pspnet_model.py         # PSPNet architecture for water segmentation
│
├── config/
│   └── config.yaml             # Master configuration (see Section 4)
│
├── weights/                    # Model weight files (.pt, .onnx)
│   ├── yolox_vivarium.pth      # YOLOX 13-class detector weights (38 MB, ACTIVE)
│   ├── water_pspnet.pt         # PSPNet water segmentation weights (178 MB)
│   ├── water_unet.pt           # UNet water segmentation weights (7.5 MB)
│   ├── food_classifier.onnx    # ONNX food classifier weights (8.7 MB)
│   └── water_classifier.onnx   # ONNX water classifier weights (8.7 MB)
│
├── train/                      # Training utilities
│   └── scripts/
│       ├── label.py            # Annotation/labelling helpers
│       ├── augment.py          # Data augmentation scripts
│       ├── split.py            # Train/val dataset split
│       ├── verify.py           # Dataset verification
│       ├── crop.py             # ROI cropping for training data
│       └── count.py            # Class distribution counter
│
├── yolox_vivarium_tiny.py      # YOLOX experiment config (13-class, tiny variant)
├── outputs/                    # Annotated output images (persisted via Docker volume)
├── logs/                       # Application and pipeline event logs
│   └── pipeline_events.jsonl   # Structured per-run JSONL audit trail
│
├── vivarium.db                 # SQLite database (auto-created, default)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 2. Architecture Overview

### Request Flow (file upload)

```
Client
  │
  ▼
POST /analyze (image + cage_id)
  │
  ├─ Validate file extension
  ├─ Read image bytes
  ├─ task_queue.enqueue()   →  generates UUID request_id
  ├─ job_tracker.set_pending(request_id)
  └─ Return { request_id, status: "PENDING" }  ← immediately (non-blocking)

[Background — asyncio worker coroutine]
  │
  ├─ job_tracker.set_processing(request_id)
  ├─ orchestrator.run(image_bytes, filename, cage_id)
  │     ├─ Deduplication check
  │     ├─ Image validation (format, size, blur)
  │     ├─ Preprocessing (resize, CLAHE)
  │     ├─ Detection (YOLOX 13-class — default; YOLOv8World / YOLOv8 switchable)
  │     ├─ Measurement (per target: water, food, mouse, bedding)
  │     │     └─ YOLOX: no second model — level decoded from class ID in bbox label
  │     ├─ Confidence gate (UNDETERMINED if below threshold)
  │     ├─ Annotation (draw boxes on image, save to outputs/)
  │     ├─ Mouse stationary check (IoU across consecutive runs)
  │     ├─ Level labelling (% → Critical/Low/OK/Full)
  │     ├─ DB persist (pipeline_results table)
  │     ├─ Alert evaluation (alert_log table if LOW/EMPTY)
  │     ├─ Webhook dispatch (fire-and-forget POST)
  │     └─ Metrics update
  └─ job_tracker.set_done(request_id, result)
       (or set_failed on exception)

Client polls:
GET /job/{request_id}  →  { status: "DONE", result: {...} }
```

### S3 Flow

Identical to the above. `POST /analyze/s3` fetches image bytes from S3 first,
then enqueues the same way. The client gets a `request_id` and polls identically.

### Singleton Lifecycle

All heavy objects are created once during FastAPI `lifespan` startup and shared
across all requests via `app.state` and module-level globals in `api/main.py`:

| Singleton | Type | Purpose |
|---|---|---|
| `_orchestrator` | `PipelineOrchestrator` | Holds loaded models |
| `_task_queue` | `TaskQueue` | asyncio queue + workers |
| `_job_tracker` | `JobTracker` | In-memory job state |
| `_metrics` | `MetricsCollector` | Runtime counters |

---

## 3. Setup & Installation

### Prerequisites

- Python 3.11+
- SQLite (zero-config default) or PostgreSQL 14+
- AWS credentials (only if using S3 image source)

### Install

```bash
git clone <repo>
cd vivarium
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env — fill in API_KEY and any DB/AWS credentials you need
```

Key variables:

| Variable | Description | Required |
|---|---|---|
| `API_KEY` | Secret key for API authentication | Yes |
| `POSTGRES_HOST` / `POSTGRES_DB` etc. | PostgreSQL connection (if not using SQLite) | No |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 access | Only for S3 source |
| `AWS_REGION` / `S3_BUCKET` | S3 region and default bucket | Only for S3 source |

### Run Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API docs available at: `http://localhost:8000/docs`

---

## 4. Configuration Reference

All configuration lives in `config/config.yaml`. Values can be overridden by environment variables
or a `config.local.yaml` for local development.

```yaml
app:
  name: "Vivarium Monitoring Pipeline"
  env: local                          # local | staging | production
  log_level: DEBUG                    # DEBUG | INFO | WARNING | ERROR

api:
  host: 0.0.0.0
  port: 8000
  cors_origins: ["*"]                 # restrict in production

# ── Targets ────────────────────────────────────────────────────────────────
targets:
  enabled: [water, food, mouse]       # which targets to detect and measure

# ── Input validation ────────────────────────────────────────────────────────
input:
  allowed_formats: [jpg, jpeg, png, webp]

# ── Task queue ──────────────────────────────────────────────────────────────
queue:
  workers: 4                          # concurrent worker coroutines
  maxsize: 100                        # max items before 503 back-pressure

# ── Database ────────────────────────────────────────────────────────────────
database:
  url: sqlite+aiosqlite:///./vivarium.db   # swap to postgresql+asyncpg://... for Postgres

# ── Detector ────────────────────────────────────────────────────────────────
detector:
  engine: yolov8world                 # yolov8world | yolov8
  weights: weights/yolov8x-worldv2.pt
  confidence: 0.25                    # detection confidence threshold
  min_confidence: 0.5                 # gate: below this → UNDETERMINED

# ── Measurers (one block per target) ────────────────────────────────────────
measurers:
  water:
    engine: fcn_psp                   # opencv_water | unet_water | fcn_psp | classifier
    min_confidence: 0.6
    weights: weights/pspnet_water.pt
  food:
    engine: opencv_food
    min_confidence: 0.5
  mouse:
    engine: detection_presence        # presence determined by detection alone

# ── Level labels ────────────────────────────────────────────────────────────
level_labels:
  water:
    - { max: 5,   label: "Empty" }
    - { max: 20,  label: "Critical" }
    - { max: 40,  label: "Low" }
    - { max: 70,  label: "OK" }
    - { max: 100, label: "Full" }
  food:
    - { max: 5,   label: "Empty" }
    - { max: 20,  label: "Critical" }
    - { max: 40,  label: "Low" }
    - { max: 70,  label: "OK" }
    - { max: 100, label: "Full" }

# ── Alerts ──────────────────────────────────────────────────────────────────
alerts:
  enabled: true
  thresholds:
    water: { low: 20.0, empty: 5.0 }
    food:  { low: 20.0, empty: 5.0 }

# ── Annotator ───────────────────────────────────────────────────────────────
annotator:
  engine: opencv
  preview_dir: outputs/               # relative to project root

# ── Deduplication ───────────────────────────────────────────────────────────
deduplication:
  enabled: true
  strategy: phash                     # phash | none
  cache_size: 64                      # max recent hashes per cage
  ttl_seconds: 300                    # hash expiry (5 minutes)
  phash_distance_threshold: 4         # max Hamming distance to flag duplicate

# ── Mouse stationary tracker ────────────────────────────────────────────────
mouse_stationary:
  enabled: true
  iou_threshold: 0.70                 # IoU >= this → same position
  consecutive_count: 2                # N same-position detections → stationary flag

# ── Webhook ─────────────────────────────────────────────────────────────────
webhook:
  enabled: false
  secret: ""                          # HMAC-SHA256 signing secret (leave blank to disable)
  timeout_seconds: 5
  urls:
    - https://your-endpoint.com/hook

# ── S3 ──────────────────────────────────────────────────────────────────────
s3:
  region: us-east-1
  bucket: ""                          # default bucket (can override per-request)
```

---

## 5. API Endpoints

### `POST /analyze` — Submit image (file upload)

Accepts a multipart image file and a `cage_id` query parameter.
Returns a `request_id` immediately. Pipeline runs in background.

**Query params:** `cage_id` (required)  
**Body:** `multipart/form-data` with `image` field  
**Response `202`:**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "PENDING",
  "message": "Job queued. Poll GET /job/550e8400-... for result."
}
```

---

### `POST /analyze/s3` — Submit image (S3 source)

Same async behaviour as `/analyze` — fetches bytes from S3 first, then enqueues.

**Body (JSON):**
```json
{ "cage_id": "cage_1", "s3_uri": "s3://my-bucket/cage_1/frame.jpg" }
```
or
```json
{ "cage_id": "cage_1", "bucket": "my-bucket", "key": "cage_1/frame.jpg" }
```
**Response `202`:** Same shape as `/analyze`.

---

### `GET /job/{request_id}` — Poll job status

**Response when PENDING/PROCESSING:**
```json
{ "request_id": "...", "cage_id": "cage_1", "status": "PENDING", "created_at": "...", "updated_at": "..." }
```

**Response when DONE:**
```json
{
  "request_id": "...",
  "status": "DONE",
  "result": {
    "cage_id": "cage_1",
    "water_pct": 67.3,
    "water_label": "OK",
    "food_pct": 12.1,
    "food_label": "Critical",
    "mouse_present": true,
    "mouse_stationary": false,
    "timestamp": "2025-01-15T10:23:45.123456+00:00",
    "success": true,
    "image_path": "outputs/abc123_frame.jpg"
  }
}
```

**Response when FAILED:**
```json
{ "request_id": "...", "status": "FAILED", "error": "Pipeline error: ..." }
```

---

### `GET /results` — Query historical results

Fetches paginated pipeline results from the database.

**Query params:**
- `cage_id` — filter by cage (optional)
- `limit` — page size (default 20)
- `offset` — pagination offset
- `from_ts` / `to_ts` — ISO timestamp range filter

---

### `GET /metrics` — Live pipeline metrics

Returns current runtime counters:
```json
{
  "total_runs": 142,
  "failed_runs": 3,
  "success_runs": 139,
  "avg_latency_ms": 234.5,
  "webhook_attempts": 139,
  "webhook_success": 137,
  "webhook_success_rate": 0.9856,
  "queue_backlog": 2
}
```

---

### `GET /health` — Health check

Used by Docker healthcheck. Returns current env, detector engine, and enabled targets.

---

## 6. Pipeline Stages (Orchestrator)

`pipeline/orchestrator.py` runs these stages in order for every job:

| # | Stage | Module | Notes |
|---|---|---|---|
| 1 | **Duplicate check** | `core/frame_deduplicator.py` | MD5 + pHash per cage. Rejected frames skip all further stages. |
| 2 | **Image validation** | `pipeline/preprocessor/image_validator.py` | Format, dimensions, corruption. Returns `rejection_reason` on failure. |
| 3 | **Preprocessing** | `pipeline/preprocessor/resizer.py` | Resize to model input size. Optional CLAHE brightness normalisation. |
| 4 | **Detection** | `pipeline/detectors/` | Returns a `BoundingBox` per target (or None if not found). |
| 5 | **Measurement** | `pipeline/measurers/` | For each detected target, crops the ROI and measures level/presence. |
| 6 | **Confidence gate** | `orchestrator.py` | If measurer confidence < `min_confidence` → label set to `UNDETERMINED`, target added to `uncertain_targets`. |
| 7 | **Annotation** | `pipeline/annotator/` | Draws boxes and labels on a copy of the image, saves to `outputs/`. |
| 8 | **Mouse stationary** | `core/mouse_stationary_tracker.py` | Compares current mouse bbox against last N runs via IoU. |
| 9 | **Level labelling** | `core/level_labeler.py` | Converts numeric % to Critical/Low/OK/Full etc. |
| 10 | **DB persistence** | `core/repositories.py` | Inserts one row into `pipeline_results`. |
| 11 | **Alert evaluation** | `core/alerting.py` | Checks water/food % against LOW/EMPTY thresholds. Inserts into `alert_log`. |
| 12 | **Webhook dispatch** | `core/webhook.py` | Fire-and-forget POST to all configured URLs. Includes alert data. |
| 13 | **Metrics update** | `core/metrics.py` | Records latency and success/failure counts. |

---

## 7. Core Modules

### `core/result.py` — Shared Data Types

All pipeline data flows through these three dataclasses. **Never define parallel schemas elsewhere.**

```python
BoundingBox        # normalised coords (0-1), confidence, label
                   # .width, .height, .area_ratio, .is_near_edge(proximity)

MeasurementResult  # level (0-100), confidence (0-1), label, present (mouse only)

PipelineResult     # full output: water_pct, food_pct, mouse_present,
                   # water_label, food_label, mouse_stationary,
                   # confidences, uncertain_targets, raw_detections,
                   # image_path, result_id, success, rejection_reason
```

### `core/config_loader.py`

Loads `config/config.yaml` (and optionally `config.local.yaml` for overrides) into a `DotMap`
for attribute-style access (`config.detector.engine`). Called once at startup and cached.

### `core/level_labeler.py`

Converts a numeric percentage to a human-readable label using ordered threshold bands from config.
First band where `pct <= max` wins. Returns `"Unknown"` if measurement is None.

```python
label_water(67.3, config)  # → "OK"
label_food(8.0, config)    # → "Critical"
```

---

## 8. Job Tracking & Task Queue

### Job Lifecycle

```
set_pending()  →  set_processing()  →  set_done(result)
                                    └→  set_failed(error)
```

### `core/job_tracker.py`

In-memory `dict[request_id → JobEntry]`. Safe for asyncio (single-thread model).

**Important:** Jobs are lost on server restart. For production, replace the backing store
with Redis or a dedicated DB table — the interface (`set_pending`, `set_done`, etc.) stays the same.

```python
tracker.set_pending(request_id, cage_id)    # called at POST /analyze
tracker.set_processing(request_id)          # called by worker
tracker.set_done(request_id, result)        # worker success
tracker.set_failed(request_id, error)       # worker exception
tracker.get(request_id) → JobEntry | None   # called by GET /job/{id}
tracker.counts() → {"PENDING": 0, ...}      # used by /metrics
```

### `core/task_queue.py`

`asyncio.Queue` with N worker coroutines. All image sources (file upload and S3) share the
same queue and worker pool.

```
POST /analyze       ──┐
POST /analyze/s3    ──┼──▶  asyncio.Queue  ──▶  worker-0
                        │                  ──▶  worker-1
                        │                  ──▶  worker-2
                        └──────────────────▶  worker-3
```

**Back-pressure:** If `queue.maxsize` is reached, new submissions return `503 Service Unavailable`.

**Shutdown:** Sends `None` sentinel values to drain workers gracefully, with a 10-second timeout per worker.

---

## 9. Database & Persistence

### `core/database.py`

SQLAlchemy async engine. Default: `sqlite+aiosqlite` (zero infrastructure).
Swap to PostgreSQL by changing `database.url` in config — no code changes needed.

Tables are created automatically on startup (`Base.metadata.create_all`).

### Tables

#### `pipeline_results`

One row per processed frame.

| Column | Type | Description |
|---|---|---|
| `id` | String (UUID) | Primary key |
| `cage_id` | String | Which cage |
| `image_path` | Text | Path to annotated output image |
| `water_pct` | Float | Water level 0–100, or NULL if uncertain |
| `food_pct` | Float | Food level 0–100, or NULL if uncertain |
| `mouse_present` | Boolean | Mouse detected |
| `water_confidence` | Float | Measurer confidence 0–1 |
| `food_confidence` | Float | Measurer confidence 0–1 |
| `mouse_confidence` | Float | Detector confidence 0–1 |
| `uncertain_targets` | JSON | List of targets that hit the UNDETERMINED gate |
| `raw_detections` | JSON | Full bounding box data for debugging |
| `processed_at` | DateTime (UTC) | When the run completed |

#### `alert_log`

One row per fired alert.

| Column | Type | Description |
|---|---|---|
| `id` | String (UUID) | Primary key |
| `cage_id` | String | Which cage triggered the alert |
| `target` | String | `"water"` or `"food"` |
| `alert_type` | String | `"LOW"` or `"EMPTY"` |
| `value` | Float | Measured percentage at time of alert |
| `message` | Text | Human-readable alert message |
| `fired_at` | DateTime (UTC) | When the alert was evaluated |

### `core/repositories.py`

All DB operations. Errors are caught and logged — a DB failure never crashes the pipeline.

```python
await insert_pipeline_result(session, result, cage_id) → result_id | None
await query_pipeline_results(session, cage_id, limit, offset, from_ts, to_ts) → list[dict]
await insert_alert(session, alert, cage_id) → alert_id | None
```

---

## 10. Alerting System

### `core/alerting.py`

After each successful pipeline run, `AlertEvaluator.evaluate()` is called.
It checks water and food percentages against two configurable thresholds:

- **EMPTY** — `value <= empty_threshold` (default 5%)
- **LOW** — `value <= low_threshold` (default 20%)

EMPTY takes priority over LOW (checked first). If neither threshold is breached, no alert is fired.

Uncertain/UNDETERMINED measurements are skipped — alerts are only fired when we have a confident reading.

Fired alerts are:
1. Written to the `alert_log` DB table
2. Returned as a list and included in the webhook payload

---

## 11. Webhooks

### `core/webhook.py`

After each pipeline run, a POST is fired to all configured URLs with this envelope:

```json
{
  "event": "pipeline.complete",
  "api_version": "1",
  "cage_id": "cage_1",
  "timestamp": "2025-01-15T10:23:45+00:00",
  "payload": { ...PipelineResult... },
  "alerts": [
    { "target": "water", "alert_type": "LOW", "value": 15.2, "message": "..." }
  ]
}
```

**Retry behaviour:** 3 attempts per URL with exponential backoff (1s → 2s → 4s).

**Signing:** When `webhook.secret` is set, an `X-Vivarium-Signature: sha256=<hex>` header is added
(HMAC-SHA256 of the raw JSON body). Receivers should verify this.

**Design note:** Webhook URLs are static (in `config.yaml`). A dynamic webhook management
endpoint is a planned future addition.

---

## 12. Metrics

### `core/metrics.py`

Thread-safe (uses `threading.Lock`) in-memory counters. Resets on server restart.

Exposed via `GET /metrics`. The `queue_backlog` counter is updated on every enqueue.

---

## 13. Detectors

### Factory Pattern

```python
# config.yaml → detector.engine: "yolov8world"
detector = DetectorFactory.create(config)
detector.load()                           # loads model weights
boxes = detector.detect(image, targets)   # → Dict[str, BoundingBox | None]
```

### Supported Engines

| Engine key | Class | Notes |
|---|---|---|
| `yolov8world` | `YOLOv8WorldDetector` | Open-vocabulary — uses text prompts, no class indices needed. Best for new categories. |
| `yolov8` | `YOLOv8Detector` | Closed-vocabulary — requires `class_map` in config mapping target names to class indices. |

### Adding a New Detector

1. Create `pipeline/detectors/my_detector.py`, subclass `BaseDetector`
2. Implement `load()` and `detect(image, targets) → Dict[str, Optional[BoundingBox]]`
3. Add to `_REGISTRY` in `pipeline/detectors/factory.py`:
   ```python
   "myengine": "pipeline.detectors.my_detector.MyDetector"
   ```
4. Set `detector.engine: myengine` in `config.yaml`

---

## 14. Measurers

### Factory Pattern

Each target (`water`, `food`, `mouse`) gets its own measurer, selected independently via
`config.measurers.<target>.engine`.

```python
measurer = MeasurerFactory.create(config, target="water")
measurer.load()
result = measurer.measure(roi_image)  # → MeasurementResult
```

### Supported Engines

| Engine key | Class | Suitable for |
|---|---|---|
| `opencv_water` | `OpenCVWaterMeasurer` | Water level via edge/contour detection |
| `opencv_food` | `OpenCVFoodMeasurer` | Food level via colour/region analysis |
| `unet_water` | `UNetWaterMeasurer` | Segmentation-based water level |
| `unet_food` | `UNetFoodMeasurer` | Segmentation-based food level |
| `fcn_psp` | `PSPNetWaterMeasurer` | PSPNet segmentation (requires PyTorch) |
| `classifier` | `ClassifierMeasurer` | ONNX classifier model |
| `detection_presence` | `DetectionPresenceMeasurer` | Mouse: present if detector fired |

### Confidence Gate (UNDETERMINED)

If a measurer returns `confidence < min_confidence` (configured per target), the measurement
label is overwritten with `"UNDETERMINED"` and the target is added to `uncertain_targets`.
The numeric `level` is still stored for debugging, but `water_pct` / `food_pct` are set to `None`
in the final result so alerts and labels don't fire on unreliable data.

---

## 15. Annotator

### `pipeline/annotator/opencv_annotator.py`

Draws bounding boxes and measurement labels on a copy of the preprocessed image and saves it to `outputs/`.

**Label placement rules (as of latest fix):**
- Label is placed **above** the bounding box by default
- If the box is near the **top edge** (no room above), the label is placed **below** the box
- Label x position is **clamped** so it never overflows the right or left image edge

**Colors:**
- Water: orange `(255, 100, 0)`
- Food: green `(0, 200, 50)`
- Mouse: yellow `(0, 220, 255)`

Output filename format: `{result_id}_{original_filename}.jpg`

---

## 16. Frame Deduplication

### `core/frame_deduplicator.py`

Prevents the same frame from being processed twice (e.g. client retries, S3 retransmits,
cameras sending identical frames at low activity).

**Two-level check per cage:**

1. **MD5 hash** — catches exact byte-for-byte duplicates
2. **Perceptual hash (pHash)** — catches visually identical frames that differ only in JPEG
   re-compression or metadata. Uses 8×8 DCT; Hamming distance ≤ threshold (default 4) = duplicate.

**Scope is per `cage_id`** — the same image for two different cages is not a duplicate.

**Cache:** Fixed-size LRU per cage (`cache_size`, default 64). Entries expire after `ttl_seconds` (default 300s).

**Strategies:**
- `phash` — full MD5 + pHash (recommended)
- `none` — disabled (pass-through)

Add new strategies via `DeduplicatorFactory.register()`.

---

## 17. Mouse Stationary Tracker

### `core/mouse_stationary_tracker.py`

Flags a mouse as stationary if it appears in the same location for `consecutive_count` (default 2)
consecutive pipeline runs. This may indicate the animal is unwell.

**Algorithm:**
1. On each run, compute IoU between current mouse bbox and the last recorded bbox for that cage
2. If `IoU >= iou_threshold` (default 0.70) → increment `consecutive_count`
3. If `consecutive_count >= consecutive_count config` → `mouse_stationary = True`
4. If IoU drops below threshold → reset count to 1
5. If mouse not detected → state is unchanged (neither incremented nor reset)

State is in-memory per cage. Use `tracker.reset(cage_id)` to clear state after an alert is handled.

---

## 18. Docker Deployment

### Build & Run

```bash
docker compose up --build
```

This builds the image, starts the container on port 8000, and mounts `./outputs` for annotated images.

### Key Docker details

- Secrets come from `.env` file (never baked into the image)
- Annotated images persist on host via volume mount: `./outputs:/app/outputs`
- Healthcheck polls `GET /health` every 30s; container marked unhealthy after 3 consecutive failures
- `start_period: 60s` — allows time for model weights to load before health checks begin
- `restart: unless-stopped` — auto-restarts on crash

### Switching to PostgreSQL in Docker

1. Add a `db` service to `docker-compose.yml`
2. Set `DATABASE_URL=postgresql+asyncpg://user:pass@db/vivarium` in `.env`
3. Update `database.url` in `config.yaml` to use the env var

---

## 19. Adding a New Engine (Extensibility Guide)

The entire project is built around the **Factory Pattern**. Swapping any component
requires only two steps: create a class, register it in the factory.

### Example: New Detector

```python
# 1. pipeline/detectors/my_detector.py
from pipeline.detectors.base import BaseDetector

class MyDetector(BaseDetector):
    def load(self):
        # load your model weights
        pass
    def detect(self, image, targets):
        # return Dict[str, Optional[BoundingBox]]
        pass

# 2. pipeline/detectors/factory.py
_REGISTRY["myengine"] = "pipeline.detectors.my_detector.MyDetector"
```

```yaml
# 3. config.yaml
detector:
  engine: myengine
```

The same pattern applies to Measurers, Annotators, and Deduplicators.

---

## 20. Known Limitations & Future Work

| Area | Current State | Recommended Improvement |
|---|---|---|
| **Job persistence** | In-memory only — lost on restart | Swap `JobTracker` backing store to Redis or a `job_status` DB table |
| **Metrics** | In-memory only — lost on restart | Export to Prometheus / Grafana |
| **Webhook management** | Static URLs in config | Add `POST /webhooks` endpoint for dynamic registration |
| **Alert deduplication** | Every run fires an alert if below threshold | Add cooldown period (e.g. max one alert per cage per hour) |
| **S3 fetch** | Synchronous boto3 `.get_object()` call blocks the event loop briefly | Wrap in `asyncio.to_thread()` or use `aioboto3` |
| **Auth** | Single global API key | Per-cage or per-user keys with role-based access |
| **Model hot-reload** | Requires server restart to change weights | Add `POST /reload` admin endpoint |
| **Tests** | Framework in place (`pytest`, `httpx`) | Expand coverage for orchestrator stages and alert logic |
