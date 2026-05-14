# Pipeline Event Log & Webhook — Implementation Notes

## 1. Pipeline Event Log (`logs/pipeline_events.jsonl`)

Every call to `POST /analyze` appends **one JSON line** to this file.

### Location
`./logs/pipeline_events.jsonl`  
Configurable via `config.yaml → logging.pipeline_log_path`.

### Record schema

| Field | Type | Description |
|---|---|---|
| `image_received_at` | ISO 8601 | Timestamp the image bytes arrived at the orchestrator |
| `image_processed_at` | ISO 8601 | Timestamp the full pipeline finished |
| `total_processing_ms` | float | Wall-clock time from receive → done (ms) |
| `cage_id` | str | Cage identifier from the client |
| `filename` | str | Original upload filename |
| `image_size_bytes` | int | Raw upload size in bytes |
| `success` | bool | Whether the pipeline ran to completion |
| `rejection_reason` | str / null | Set when image was rejected at preprocessor |
| `uncertain_targets` | list[str] | Targets with no detection / measurement |
| `water_pct` | float / null | Water fill level 0–100 |
| `water_label` | str / null | Human label (Critical / Low / OK / Full) |
| `water_confidence` | float / null | Detector + measurer confidence 0–1 |
| `water_bbox` | object / null | {x1, y1, x2, y2} normalised 0–1 |
| `food_pct` | float / null | Food fill level 0–100 |
| `food_label` | str / null | Human label |
| `food_confidence` | float / null | |
| `food_bbox` | object / null | {x1, y1, x2, y2} normalised 0–1 |
| `mouse_present` | bool / null | Whether mouse was detected |
| `mouse_stationary` | bool / null | Stationary flag from tracker |
| `mouse_confidence` | float / null | |
| `mouse_bbox` | object / null | {x1, y1, x2, y2} normalised 0–1 |
| `mouse_previous_bbox` | object / null | Bbox stored by tracker BEFORE this run |
| `webhook_fired` | bool | True if at least one webhook POST succeeded |

### Example record
{"image_received_at":"2025-01-15T10:23:44.810+00:00","image_processed_at":"2025-01-15T10:23:45.123+00:00","total_processing_ms":313.0,"cage_id":"cage_1","filename":"frame_001.jpg","image_size_bytes":204800,"success":true,"rejection_reason":null,"uncertain_targets":[],"water_pct":62.5,"water_label":"OK","water_confidence":0.912,"water_bbox":{"x1":0.12,"y1":0.08,"x2":0.45,"y2":0.91},"food_pct":25.0,"food_label":"Low","food_confidence":0.801,"food_bbox":{"x1":0.55,"y1":0.60,"x2":0.88,"y2":0.95},"mouse_present":true,"mouse_stationary":false,"mouse_confidence":0.974,"mouse_bbox":{"x1":0.30,"y1":0.20,"x2":0.65,"y2":0.75},"mouse_previous_bbox":{"x1":0.28,"y1":0.19,"x2":0.63,"y2":0.74},"webhook_fired":true}

---

## 2. Webhook Dispatcher (`core/webhook.py`)

### Enabling
Edit config.yaml:

    webhook:
      enabled: true
      secret: "my-signing-secret"   # optional
      timeout_seconds: 5
      urls:
        - https://example.com/vivarium-hook
        - https://n8n.yourdomain.com/webhook/abc123

Restart the server. No DB or UI changes needed.

### Payload envelope

    {
      "event":       "pipeline.complete",
      "api_version": "1",
      "cage_id":     "cage_1",
      "timestamp":   "2025-01-15T10:23:45.200+00:00",
      "payload":     { ...full PipelineResult... }
    }

### Signature verification (when secret is set)

Header sent:  X-Vivarium-Signature: sha256=<hmac-sha256-hex>

Python receiver check:
    import hashlib, hmac
    def verify(body, header, secret):
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, header)

### Behaviour
- Fire-and-forget: a slow/dead endpoint never blocks /analyze response
- Failures are logged at WARNING/ERROR, never raised
- webhook_fired in event log = True if at least one URL returned 2xx
- To test temporarily: add https://webhook.site/your-id to urls and restart
