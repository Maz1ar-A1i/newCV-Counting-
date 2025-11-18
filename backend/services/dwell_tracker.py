from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "face_recognition library is required for dwell tracking. "
        "Make sure it is included in requirements.txt."
    ) from exc

from sqlalchemy.orm import Session

from backend.zone_logic import ZoneSnapshot, point_in_polygon
from ..models import DwellTarget, DwellTargetSession


@dataclass
class TargetProfile:
    target_id: str
    name: str
    camera_id: int
    zone_ids: List[str]
    encoding: np.ndarray
    match_threshold: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ActiveZoneState:
    zone_id: str
    inside: bool = False
    session_id: Optional[str] = None
    entry_ts: Optional[datetime] = None
    last_seen: Optional[datetime] = None


class DwellTrackingEngine:
    """
    Performs face-based tracking for configured dwell targets and
    persists dwell sessions for specific zones.
    """

    def __init__(
        self,
        session_factory,
        zone_cache,
        cache_ttl_seconds: int = 10,
        lost_timeout_ms: int = 1800,
    ):
        self.session_factory = session_factory
        self.zone_cache = zone_cache
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.lost_timeout = timedelta(milliseconds=lost_timeout_ms)
        self.targets_by_camera: Dict[int, List[TargetProfile]] = {}
        self.target_state: Dict[str, Dict[str, ActiveZoneState]] = {}
        self._lock = RLock()
        self._last_sync = datetime.min

    # -------------- public helpers --------------
    def invalidate_cache(self) -> None:
        with self._lock:
            self._last_sync = datetime.min

    def process_frame(self, camera_id: int, frame, timestamp: Optional[datetime] = None) -> None:
        if timestamp is None:
            timestamp = datetime.utcnow()

        with self._lock:
            self._refresh_targets_if_needed(force=False)
            targets = self.targets_by_camera.get(camera_id, [])
            if not targets:
                self._handle_timeouts(camera_id, timestamp)
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            if not face_locations:
                self._handle_timeouts(camera_id, timestamp)
                return

            encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            detections = list(zip(face_locations, encodings))
            zone_map = self._zone_map_for_camera(camera_id)

            for (top, right, bottom, left), encoding in detections:
                centroid = ((left + right) / 2.0, (top + bottom) / 2.0)
                best_match: Optional[Tuple[TargetProfile, float]] = None

                for profile in targets:
                    distance = np.linalg.norm(profile.encoding - encoding)
                    if distance <= profile.match_threshold:
                        if best_match is None or distance < best_match[1]:
                            best_match = (profile, distance)

                if not best_match:
                    continue

                profile = best_match[0]
                self._update_target_presence(
                    profile=profile,
                    centroid=centroid,
                    timestamp=timestamp,
                    zone_map=zone_map,
                )

            self._handle_timeouts(camera_id, timestamp)

    def live_sessions(self) -> List[Dict]:
        with self._lock:
            payload: List[Dict] = []
            for target_id, zones in self.target_state.items():
                for zone_state in zones.values():
                    if zone_state.inside and zone_state.entry_ts:
                        payload.append(
                            {
                                "target_id": target_id,
                                "zone_id": zone_state.zone_id,
                                "session_id": zone_state.session_id,
                                "entry_ts": zone_state.entry_ts.isoformat(),
                                "dwell_seconds": max(
                                    (datetime.utcnow() - zone_state.entry_ts).total_seconds(), 0
                                ),
                            }
                        )
            return payload

    # -------------- internal helpers --------------
    def _refresh_targets_if_needed(self, force: bool = False) -> None:
        if not force and datetime.utcnow() - self._last_sync < self.cache_ttl:
            return

        session: Session = self.session_factory()
        try:
            records = (
                session.query(DwellTarget)
                .filter(DwellTarget.is_active.is_(True))
                .all()
            )
            new_map: Dict[int, List[TargetProfile]] = {}
            for record in records:
                if not record.zone_ids:
                    continue
                encoding = np.array(record.face_encoding, dtype=np.float32)
                profile = TargetProfile(
                    target_id=str(record.target_id),
                    name=record.name,
                    camera_id=record.camera_id,
                    zone_ids=record.zone_ids,
                    encoding=encoding,
                    match_threshold=record.match_threshold or 0.45,
                    metadata=record.extra_data or {},
                )
                new_map.setdefault(record.camera_id, []).append(profile)

            self.targets_by_camera = new_map
            self._last_sync = datetime.utcnow()
        finally:
            session.close()

    def _zone_map_for_camera(self, camera_id: int) -> Dict[str, ZoneSnapshot]:
        zones = self.zone_cache.get_for_camera(camera_id)
        return {zone.zone_id: zone for zone in zones}

    def _update_target_presence(
        self,
        profile: TargetProfile,
        centroid: Tuple[float, float],
        timestamp: datetime,
        zone_map: Dict[str, ZoneSnapshot],
    ) -> None:
        state = self.target_state.setdefault(profile.target_id, {})

        for zone_id in profile.zone_ids:
            zone = zone_map.get(zone_id)
            if not zone:
                continue
            inside = point_in_polygon(centroid, zone.polygon)
            zone_state = state.setdefault(zone_id, ActiveZoneState(zone_id=zone_id))
            zone_state.last_seen = timestamp

            if inside and not zone_state.inside:
                session_id = self._start_session(profile, zone_id, timestamp)
                zone_state.inside = True
                zone_state.entry_ts = timestamp
                zone_state.session_id = session_id
            elif not inside and zone_state.inside:
                self._close_session(profile, zone_state, timestamp)

    def _start_session(self, profile: TargetProfile, zone_id: str, entry_ts: datetime) -> Optional[str]:
        session = self.session_factory()
        try:
            record = DwellTargetSession(
                target_id=profile.target_id,
                zone_id=zone_id,
                camera_id=profile.camera_id,
                entry_ts=entry_ts,
                status="active",
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return str(record.session_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _close_session(
        self,
        profile: TargetProfile,
        zone_state: ActiveZoneState,
        timestamp: datetime,
    ) -> None:
        if not zone_state.session_id or not zone_state.entry_ts:
            zone_state.inside = False
            zone_state.session_id = None
            zone_state.entry_ts = None
            return

        session = self.session_factory()
        try:
            dwell_seconds = max((timestamp - zone_state.entry_ts).total_seconds(), 0)
            (
                session.query(DwellTargetSession)
                .filter(DwellTargetSession.session_id == zone_state.session_id)
                .update(
                    {
                        "exit_ts": timestamp,
                        "dwell_seconds": dwell_seconds,
                        "status": "completed",
                    }
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

        zone_state.inside = False
        zone_state.session_id = None
        zone_state.entry_ts = None

    def _handle_timeouts(self, camera_id: int, timestamp: datetime) -> None:
        """
        Close sessions that have not been updated recently.
        """
        for target_id, zones in list(self.target_state.items()):
            for zone_state in zones.values():
                if zone_state.inside and zone_state.last_seen:
                    if timestamp - zone_state.last_seen > self.lost_timeout:
                        profile = self._find_profile(target_id, camera_id)
                        if profile:
                            self._close_session(profile, zone_state, timestamp)

    def _find_profile(self, target_id: str, camera_id: int) -> Optional[TargetProfile]:
        for profile in self.targets_by_camera.get(camera_id, []):
            if profile.target_id == target_id:
                return profile
        return None

