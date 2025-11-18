from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, List, Optional, Set, Tuple, Literal, Any
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from .models import Zone, DwellEvent, ZoneCounterEvent


Point = Tuple[float, float]
Polygon = List[Point]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """
    Ray casting algorithm to determine whether a point lies within a polygon.
    Assumes polygon vertices are specified as [[x, y], ...].
    """
    if len(polygon) < 3:
        return False

    x, y = point
    inside = False
    num_points = len(polygon)

    for i in range(num_points):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % num_points]

        # Check if point is on an horizontal boundary
        if y1 == y2 and y == y1 and min(x1, x2) <= x <= max(x1, x2):
            return True

        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
        )
        if intersects:
            inside = not inside
    return inside


def hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    color_hex = color_hex.lstrip("#")
    if len(color_hex) != 6:
        return 0, 0, 255
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return b, g, r


@dataclass
class ZoneSnapshot:
    zone_id: str
    camera_id: int
    name: str
    polygon: List[List[float]]
    color: str
    attribution_mode: str
    is_deleted: bool
    properties: Dict

    @classmethod
    def from_model(cls, zone: Zone) -> "ZoneSnapshot":
        return cls(
            zone_id=zone.zone_id,
            camera_id=zone.camera_id,
            name=zone.name,
            polygon=zone.polygon,
            color=zone.color,
            attribution_mode=zone.attribution_mode,
            is_deleted=zone.is_deleted,
            properties=zone.properties or {},
        )


@dataclass
class ZoneAction:
    action: Literal["create", "update", "delete"]
    before: Optional[ZoneSnapshot]
    after: Optional[ZoneSnapshot]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UndoRedoManager:
    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
        self._history: List[ZoneAction] = []
        self._future: List[ZoneAction] = []
        self._lock = RLock()

    def record(self, action: ZoneAction) -> None:
        with self._lock:
            self._history.append(action)
            if len(self._history) > self.max_depth:
                self._history.pop(0)
            self._future.clear()

    def undo(self) -> Optional[ZoneAction]:
        with self._lock:
            if not self._history:
                return None
            action = self._history.pop()
            self._future.append(action)
            return action

    def redo(self) -> Optional[ZoneAction]:
        with self._lock:
            if not self._future:
                return None
            action = self._future.pop()
            self._history.append(action)
            return action


class ZoneCache:
    def __init__(self):
        self._zones_by_camera: Dict[int, List[ZoneSnapshot]] = {}
        self._lock = RLock()

    def refresh(self, session_factory) -> None:
        session: Session = session_factory()
        try:
            zones = (
                session.query(Zone)
                .filter(Zone.is_deleted.is_(False))
                .order_by(Zone.camera_id, Zone.created_at)
                .all()
            )
        finally:
            session.close()

        zones_by_camera: Dict[int, List[ZoneSnapshot]] = {}
        for zone in zones:
            zones_by_camera.setdefault(zone.camera_id, []).append(ZoneSnapshot.from_model(zone))

        with self._lock:
            self._zones_by_camera = zones_by_camera

    def get_for_camera(self, camera_id: int) -> List[ZoneSnapshot]:
        with self._lock:
            return list(self._zones_by_camera.get(camera_id, []))


def _centroid_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


class CentroidTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_object_id = 1
        self.objects: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, detection: Dict) -> int:
        object_id = self.next_object_id
        self.next_object_id += 1
        self.objects[object_id] = detection
        self.disappeared[object_id] = 0
        return object_id

    def deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(
        self,
        detections: List[Dict],
        timestamp: datetime,
    ) -> Tuple[Dict[int, Dict], List[int]]:
        """
        Update tracker with new detections.
        Each detection dict must contain 'bbox', 'centroid', 'confidence', 'class_name'.
        Returns tuple (active_objects, lost_ids).
        """
        lost_ids: List[int] = []

        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    lost_ids.append(object_id)
                    self.deregister(object_id)
            return self.objects.copy(), lost_ids

        if len(self.objects) == 0:
            for det in detections:
                det["timestamp"] = timestamp
                self.register(det)
            return self.objects.copy(), lost_ids

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[obj_id]["centroid"] for obj_id in object_ids])
        input_centroids = np.array([det["centroid"] for det in detections])

        distances = np.linalg.norm(object_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if distances[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]
            detection = detections[col]
            detection["timestamp"] = timestamp
            self.objects[object_id] = detection
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
        unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

        if distances.shape[0] >= distances.shape[1]:
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    lost_ids.append(object_id)
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                detection = detections[col]
                detection["timestamp"] = timestamp
                self.register(detection)

        return self.objects.copy(), lost_ids


@dataclass
class ActiveDwell:
    event_id: Optional[UUID]
    entry_ts: datetime
    object_type: str
    tracker_object_id: int
    object_label: str


class ZonePresenceEngine:
    """
    Tracks object-zone relationships, handles dwell event creation,
    and exposes live presence counters.
    """

    def __init__(
        self,
        session_factory,
        zone_cache: ZoneCache,
        debounce_ms: int = 300,
        lost_track_timeout_ms: int = 1200,
    ):
        self.session_factory = session_factory
        self.zone_cache = zone_cache
        self.debounce = timedelta(milliseconds=debounce_ms)
        self.lost_timeout = timedelta(milliseconds=lost_track_timeout_ms)
        self.trackers: Dict[int, CentroidTracker] = {}
        self.object_states: Dict[int, Dict[int, Dict]] = {}
        self.live_counts: Dict[str, Dict] = {}
        self._lock = RLock()

    def _get_tracker(self, camera_id: int) -> CentroidTracker:
        if camera_id not in self.trackers:
            self.trackers[camera_id] = CentroidTracker()
        return self.trackers[camera_id]

    def _ensure_state(self, camera_id: int, object_id: int) -> Dict:
        self.object_states.setdefault(camera_id, {})
        self.object_states[camera_id].setdefault(
            object_id,
            {
                "current_zones": set(),
                "pending_entries": {},
                "pending_exits": {},
                "active_events": {},
                "last_seen": datetime.utcnow(),
                "last_observation": None,
            },
        )
        return self.object_states[camera_id][object_id]

    def process_detections(
        self,
        camera_id: int,
        detections: List[Dict],
        timestamp: Optional[datetime] = None,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.utcnow()

        normalized_detections: List[Dict] = []
        for det in detections:
            bbox = det.get("bbox")
            if not bbox:
                continue
            normalized_detections.append(
                {
                    "bbox": bbox,
                    "centroid": det.get("centroid") or _centroid_from_bbox(bbox),
                    "confidence": det.get("confidence", 0.0),
                    "class_name": det.get("class_name", "unknown"),
                }
            )

        if not normalized_detections and camera_id not in self.trackers:
            return

        with self._lock:
            tracker = self._get_tracker(camera_id)
            active_objects, lost_ids = tracker.update(normalized_detections, timestamp)

            for object_id, observation in active_objects.items():
                self._update_object_state(
                    camera_id=camera_id,
                    object_id=object_id,
                    observation=observation,
                    timestamp=timestamp,
                )

            for lost_id in lost_ids:
                self._force_exit(camera_id, lost_id, timestamp)

    def _update_object_state(
        self,
        camera_id: int,
        object_id: int,
        observation: Dict,
        timestamp: datetime,
    ) -> None:
        zones = self.zone_cache.get_for_camera(camera_id)
        if not zones:
            return

        centroid = observation.get("centroid")
        if centroid is None:
            centroid = _centroid_from_bbox(observation["bbox"])

        state = self._ensure_state(camera_id, object_id)
        state["last_observation"] = observation
        current_zones = {zone.zone_id for zone in zones if point_in_polygon(centroid, zone.polygon)}
        previous_zones = state["current_zones"]

        entered = current_zones - previous_zones
        exited = previous_zones - current_zones

        # handle debounce for entries
        for zone_id in entered:
            pending_since = state["pending_entries"].get(zone_id)
            if pending_since is None:
                state["pending_entries"][zone_id] = timestamp
                continue
            if timestamp - pending_since >= self.debounce:
                self._confirm_entry(camera_id, object_id, zone_id, timestamp, observation)

        for zone_id in exited:
            pending_since = state["pending_exits"].get(zone_id)
            if pending_since is None:
                state["pending_exits"][zone_id] = timestamp
                continue
            if timestamp - pending_since >= self.debounce:
                self._confirm_exit(camera_id, object_id, zone_id, timestamp)

        # Clear pending states for zones we're firmly inside/outside
        for zone_id in list(state["pending_entries"].keys()):
            if zone_id not in current_zones:
                state["pending_entries"].pop(zone_id, None)

        for zone_id in list(state["pending_exits"].keys()):
            if zone_id in current_zones:
                state["pending_exits"].pop(zone_id, None)

        state["current_zones"] = current_zones
        state["last_seen"] = timestamp

    def _confirm_entry(
        self,
        camera_id: int,
        object_id: int,
        zone_id: str,
        timestamp: datetime,
        observation: Optional[Dict] = None,
    ) -> None:
        state = self._ensure_state(camera_id, object_id)
        if zone_id in state["active_events"]:
            return

        observation = observation or state.get("last_observation") or {}
        object_type = observation.get("class_name", "unknown")
        object_label = f"{camera_id}:{object_id}"

        session = self.session_factory()
        try:
            event = DwellEvent(
                zone_id=zone_id,
                camera_id=camera_id,
                object_id=object_label,
                object_type=object_type,
                entry_ts=timestamp,
                extra_data={
                    "confidence": observation.get("confidence"),
                    "bbox": observation.get("bbox"),
                },
            )
            session.add(event)
            session.flush()
            state["active_events"][zone_id] = ActiveDwell(
                event_id=event.event_id,
                entry_ts=timestamp,
                object_type=object_type,
                tracker_object_id=object_id,
                object_label=object_label,
            )
            state["pending_entries"].pop(zone_id, None)
            state["current_zones"].add(zone_id)
            self._increment_live(zone_id, event.event_id, timestamp, object_id, object_type)
            self._log_zone_event(
                session=session,
                zone_id=zone_id,
                camera_id=camera_id,
                object_label=object_label,
                object_type=object_type,
                event_type="enter",
                timestamp=timestamp,
                dwell_event_id=event.event_id,
            )
            session.commit()
            session.refresh(event)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _confirm_exit(self, camera_id: int, object_id: int, zone_id: str, timestamp: datetime) -> None:
        state = self._ensure_state(camera_id, object_id)
        active_event = state["active_events"].get(zone_id)
        if not active_event:
            state["pending_exits"].pop(zone_id, None)
            state["current_zones"].discard(zone_id)
            return

        session = self.session_factory()
        try:
            dwell_seconds = max((timestamp - active_event.entry_ts).total_seconds(), 0)
            (
                session.query(DwellEvent)
                .filter(DwellEvent.event_id == active_event.event_id)
                .update({"exit_ts": timestamp, "dwell_seconds": dwell_seconds})
            )
            self._log_zone_event(
                session=session,
                zone_id=zone_id,
                camera_id=camera_id,
                object_label=active_event.object_label,
                object_type=active_event.object_type,
                event_type="exit",
                timestamp=timestamp,
                dwell_event_id=active_event.event_id,
            )
            session.commit()
            state["active_events"].pop(zone_id, None)
            state["pending_exits"].pop(zone_id, None)
            state["current_zones"].discard(zone_id)
            self._decrement_live(zone_id, active_event.event_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _increment_live(
        self,
        zone_id: str,
        event_id,
        timestamp: datetime,
        object_id: int,
        object_type: str,
    ) -> None:
        zone_state = self.live_counts.setdefault(zone_id, {"count": 0, "objects": {}})
        zone_state["count"] += 1
        zone_state["objects"][str(event_id)] = {
            "entry_ts": timestamp,
            "object_id": object_id,
            "object_type": object_type,
        }

    def _decrement_live(self, zone_id: str, event_id) -> None:
        zone_state = self.live_counts.setdefault(zone_id, {"count": 0, "objects": {}})
        if zone_state["count"] > 0:
            zone_state["count"] -= 1
        zone_state["objects"].pop(str(event_id), None)

    def _log_zone_event(
        self,
        session: Session,
        zone_id: str,
        camera_id: int,
        object_label: str,
        object_type: str,
        event_type: str,
        timestamp: datetime,
        dwell_event_id,
    ) -> None:
        zone_event = ZoneCounterEvent(
            zone_id=zone_id,
            camera_id=camera_id,
            object_id=object_label,
            object_type=object_type,
            event_type=event_type,
            timestamp=timestamp,
            dwell_event_id=dwell_event_id,
        )
        session.add(zone_event)

    def _force_exit(self, camera_id: int, object_id: int, timestamp: datetime) -> None:
        state = self.object_states.get(camera_id, {}).get(object_id)
        if not state:
            return

        for zone_id in list(state["current_zones"]):
            self._confirm_exit(camera_id, object_id, zone_id, timestamp)

        self.object_states[camera_id].pop(object_id, None)

    def live_snapshot(self) -> Dict[str, Dict]:
        with self._lock:
            snapshot = {}
            for zone_id, info in self.live_counts.items():
                objects = []
                by_type: Dict[str, Dict] = {}
                for event_id, meta in info["objects"].items():
                    dwell_seconds = max((datetime.utcnow() - meta["entry_ts"]).total_seconds(), 0)
                    payload = {
                        "event_id": event_id,
                        "object_id": meta["object_id"],
                        "object_type": meta.get("object_type", "unknown"),
                        "dwell_seconds": dwell_seconds,
                        "entry_ts": meta["entry_ts"].isoformat(),
                    }
                    objects.append(payload)
                    bucket = by_type.setdefault(
                        payload["object_type"],
                        {"count": 0, "objects": []},
                    )
                    bucket["count"] += 1
                    bucket["objects"].append(payload)
                snapshot[zone_id] = {"count": info["count"], "objects": objects, "by_type": by_type}
            return snapshot

    def drop_zone(self, zone_id: str) -> None:
        with self._lock:
            now = datetime.utcnow()
            # Close active events referencing this zone
            for camera_id, camera_states in list(self.object_states.items()):
                for object_id, state in list(camera_states.items()):
                    if zone_id in state["current_zones"]:
                        self._confirm_exit(camera_id, object_id, zone_id, now)
                    state["pending_entries"].pop(zone_id, None)
                    state["pending_exits"].pop(zone_id, None)
                    state["active_events"].pop(zone_id, None)
            self.live_counts.pop(zone_id, None)

