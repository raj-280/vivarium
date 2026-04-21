-- migrations/001_initial.sql
-- Initial schema for the Vivarium Monitoring Pipeline
-- Run once against the target PostgreSQL database.

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =========================================================================
-- Table: pipeline_results
-- Stores the output of every complete pipeline run.
-- =========================================================================
CREATE TABLE IF NOT EXISTS pipeline_results (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_path       TEXT,
    water_pct        FLOAT,
    food_pct         FLOAT,
    mouse_present    BOOLEAN,
    water_confidence FLOAT,
    food_confidence  FLOAT,
    mouse_confidence FLOAT,
    uncertain_targets TEXT[]    DEFAULT '{}',
    raw_detections   JSONB      DEFAULT '{}',
    processed_at     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_results_processed_at
    ON pipeline_results (processed_at DESC);

CREATE INDEX IF NOT EXISTS idx_pipeline_results_water_pct
    ON pipeline_results (water_pct)
    WHERE water_pct IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_results_food_pct
    ON pipeline_results (food_pct)
    WHERE food_pct IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pipeline_results_mouse_present
    ON pipeline_results (mouse_present)
    WHERE mouse_present IS NOT NULL;

-- =========================================================================
-- Table: alert_log
-- Audit log of every fired alert.
-- =========================================================================
CREATE TABLE IF NOT EXISTS alert_log (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target           TEXT             NOT NULL,  -- water | food | mouse | image
    alert_type       TEXT             NOT NULL,  -- low | missing | rejected
    value            FLOAT,
    message          TEXT             NOT NULL,
    notifiers_fired  TEXT[]           DEFAULT '{}',
    fired_at         TIMESTAMPTZ      DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alert_log_fired_at
    ON alert_log (fired_at DESC);

CREATE INDEX IF NOT EXISTS idx_alert_log_target
    ON alert_log (target);

-- =========================================================================
-- Table: cooldown_state
-- Single row per target — tracks when the last alert was fired.
-- Used by CooldownManager to gate repeated notifications.
-- =========================================================================
CREATE TABLE IF NOT EXISTS cooldown_state (
    target           TEXT PRIMARY KEY,
    last_alert_at    TIMESTAMPTZ NOT NULL
);

-- Seed empty cooldown rows for known targets (harmless if re-run)
INSERT INTO cooldown_state (target, last_alert_at)
    VALUES
        ('water', '2000-01-01 00:00:00+00'),
        ('food',  '2000-01-01 00:00:00+00'),
        ('mouse', '2000-01-01 00:00:00+00'),
        ('image', '2000-01-01 00:00:00+00')
    ON CONFLICT (target) DO NOTHING;
