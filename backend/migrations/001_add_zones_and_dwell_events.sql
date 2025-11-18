BEGIN;

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS zones (
    zone_id VARCHAR PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    name VARCHAR NOT NULL,
    polygon JSONB NOT NULL,
    color VARCHAR DEFAULT '#FF0000',
    attribution_mode VARCHAR DEFAULT 'multiple',
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    properties JSONB DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_zone_camera_name ON zones(camera_id, name);

CREATE TABLE IF NOT EXISTS dwell_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id VARCHAR REFERENCES zones(zone_id) ON DELETE SET NULL,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    object_id VARCHAR NOT NULL,
    entry_ts TIMESTAMPTZ NOT NULL,
    exit_ts TIMESTAMPTZ NULL,
    dwell_seconds DOUBLE PRECISION NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_dwell_events_zone_open ON dwell_events (zone_id, exit_ts);
CREATE INDEX IF NOT EXISTS ix_dwell_events_object ON dwell_events (object_id);

COMMIT;

