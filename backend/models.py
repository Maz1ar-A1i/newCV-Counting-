from datetime import datetime
import uuid

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    class_name = Column(String)
    camera_id = Column(Integer, index=True)
    camera_name = Column(String)


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String, nullable=False, unique=True)
    stream_type = Column(String)
    stream = Column(String)
    location = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __init__(self, source_name, stream_type, stream, location=None):
        self.source_name = source_name
        self.stream_type = stream_type
        self.stream = stream
        self.location = location
        self.created_at = datetime.utcnow()


class Zone(Base):
    __tablename__ = "zones"

    zone_id = Column(String, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), index=True, nullable=False)
    name = Column(String, nullable=False)
    polygon = Column(JSONB, nullable=False)
    color = Column(String, default="#FF0000")
    attribution_mode = Column(String, default="multiple")
    is_deleted = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    properties = Column(JSONB, default=dict)

    __table_args__ = (
        UniqueConstraint("camera_id", "name", name="uq_zone_camera_name"),
    )


class DwellEvent(Base):
    __tablename__ = "dwell_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id = Column(String, ForeignKey("zones.zone_id", ondelete="SET NULL"), index=True, nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), index=True, nullable=False)
    object_id = Column(String, index=True, nullable=False)
    object_type = Column(String, nullable=False, default="unknown", index=True)
    entry_ts = Column(DateTime, nullable=False, index=True)
    exit_ts = Column(DateTime, nullable=True, index=True)
    dwell_seconds = Column(Float, nullable=True)
    extra_data = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_dwell_events_zone_open", "zone_id", "exit_ts"),
    )


class DwellTarget(Base):
    __tablename__ = "dwell_targets"

    target_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), index=True, nullable=False)
    zone_ids = Column(JSONB, default=list)
    face_encoding = Column(JSONB, nullable=False)
    match_threshold = Column(Float, default=0.45)
    reference_image_path = Column(String, nullable=True)
    extra_data = Column(JSONB, default=dict)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DwellTargetSession(Base):
    __tablename__ = "dwell_target_sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_id = Column(UUID(as_uuid=True), ForeignKey("dwell_targets.target_id", ondelete="CASCADE"), index=True, nullable=False)
    zone_id = Column(String, ForeignKey("zones.zone_id", ondelete="SET NULL"), index=True, nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), index=True, nullable=False)
    entry_ts = Column(DateTime, nullable=False, index=True)
    exit_ts = Column(DateTime, nullable=True, index=True)
    dwell_seconds = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="active", index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ZoneCounterEvent(Base):
    __tablename__ = "zone_counter_events"

    event_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id = Column(String, ForeignKey("zones.zone_id", ondelete="SET NULL"), index=True, nullable=True)
    camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), index=True, nullable=False)
    object_id = Column(String, nullable=False, index=True)
    object_type = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)  # enter | exit
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    dwell_event_id = Column(UUID(as_uuid=True), ForeignKey("dwell_events.event_id", ondelete="SET NULL"), nullable=True)
    extra_data = Column(JSONB, default=dict)