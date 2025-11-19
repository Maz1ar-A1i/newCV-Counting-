from datetime import datetime
from typing import Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, ConfigDict


class CameraCreate(BaseModel):
    source_name: str
    stream_type: str
    stream: str
    location: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PolygonPoint(BaseModel):
    x: float
    y: float

    def as_list(self) -> List[float]:
        return [self.x, self.y]


class ZoneBase(BaseModel):
    camera_id: int
    name: str
    polygon: List[List[float]]
    color: str = "#FF0000"
    attribution_mode: Literal["multiple", "exclusive"] = "multiple"
    properties: Dict = Field(default_factory=dict)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, value):
        if len(value) < 3:
            raise ValueError("Polygon must contain at least 3 points")
        return value


class ZoneCreate(ZoneBase):
    zone_id: Optional[str] = None


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    polygon: Optional[List[List[float]]] = None
    color: Optional[str] = None
    attribution_mode: Optional[Literal["multiple", "exclusive"]] = None
    properties: Optional[Dict] = None


class ZoneResponse(ZoneBase):
    zone_id: str
    is_deleted: bool = False
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)


class DwellEventResponse(BaseModel):
    event_id: UUID
    zone_id: Optional[str]
    camera_id: int
    object_id: str
    object_type: str
    entry_ts: datetime
    exit_ts: Optional[datetime]
    dwell_seconds: Optional[float]

    model_config = ConfigDict(from_attributes=True)


class DwellTargetResponse(BaseModel):
    target_id: UUID
    name: str
    camera_id: int
    zone_ids: List[str]
    match_threshold: float
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DwellTargetUpdate(BaseModel):
    name: Optional[str] = None
    zone_ids: Optional[List[str]] = None
    match_threshold: Optional[float] = None
    is_active: Optional[bool] = None


class DwellSessionResponse(BaseModel):
    session_id: UUID
    target_id: UUID
    zone_id: Optional[str]
    camera_id: int
    entry_ts: datetime
    exit_ts: Optional[datetime]
    dwell_seconds: Optional[float]
    status: str

    model_config = ConfigDict(from_attributes=True)


class ZoneCounterEventResponse(BaseModel):
    event_id: UUID
    zone_id: Optional[str]
    camera_id: int
    object_id: str
    object_type: str
    event_type: str
    timestamp: datetime

    model_config = ConfigDict(from_attributes=True)