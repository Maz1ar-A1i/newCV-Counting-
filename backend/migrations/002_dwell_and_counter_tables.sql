BEGIN;

ALTER TABLE dwell_events
    ADD COLUMN IF NOT EXISTS object_type VARCHAR DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS extra_data JSONB DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS dwell_targets (
    target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR NOT NULL,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    zone_ids JSONB DEFAULT '[]'::jsonb,
    face_encoding JSONB NOT NULL,
    match_threshold DOUBLE PRECISION DEFAULT 0.45,
    reference_image_path VARCHAR NULL,
    extra_data JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_dwell_targets_camera ON dwell_targets(camera_id);
CREATE INDEX IF NOT EXISTS ix_dwell_targets_active ON dwell_targets(is_active);

CREATE TABLE IF NOT EXISTS dwell_target_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID NOT NULL REFERENCES dwell_targets(target_id) ON DELETE CASCADE,
    zone_id VARCHAR REFERENCES zones(zone_id) ON DELETE SET NULL,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    entry_ts TIMESTAMPTZ NOT NULL,
    exit_ts TIMESTAMPTZ NULL,
    dwell_seconds DOUBLE PRECISION NULL,
    status VARCHAR DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_dwell_target_sessions_target ON dwell_target_sessions(target_id);
CREATE INDEX IF NOT EXISTS ix_dwell_target_sessions_zone ON dwell_target_sessions(zone_id);
CREATE INDEX IF NOT EXISTS ix_dwell_target_sessions_status ON dwell_target_sessions(status);

CREATE TABLE IF NOT EXISTS zone_counter_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_id VARCHAR REFERENCES zones(zone_id) ON DELETE SET NULL,
    camera_id INTEGER NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    object_id VARCHAR NOT NULL,
    object_type VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    dwell_event_id UUID REFERENCES dwell_events(event_id) ON DELETE SET NULL,
    extra_data JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_zone_counter_events_zone ON zone_counter_events(zone_id);
CREATE INDEX IF NOT EXISTS ix_zone_counter_events_type ON zone_counter_events(object_type);
CREATE INDEX IF NOT EXISTS ix_zone_counter_events_event_type ON zone_counter_events(event_type);

COMMIT;

