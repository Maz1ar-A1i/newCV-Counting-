from datetime import datetime, timedelta
import uuid

from backend.zone_logic import (
    point_in_polygon,
    ZoneCache,
    ZoneSnapshot,
    ZonePresenceEngine,
    UndoRedoManager,
    ZoneAction,
)


class MemorySession:
    def __init__(self):
        self.events = {}
        self.updated = {}

    def add(self, event):
        if getattr(event, "event_id", None) is None:
            event.event_id = uuid.uuid4()
        self.events[event.event_id] = event

    def commit(self):
        return True

    def rollback(self):
        return False

    def refresh(self, event):
        return event

    def close(self):
        return None

    def query(self, model):
        return MemoryQuery(self)


class MemoryQuery:
    def __init__(self, session: MemorySession):
        self.session = session
        self.event_id = None

    def filter(self, criterion):
        # Expect SQLAlchemy BinaryExpression
        if hasattr(criterion, "right"):
            self.event_id = getattr(criterion.right, "value", None)
        else:
            self.event_id = criterion
        return self

    def update(self, values):
        event = self.session.events.get(self.event_id)
        if not event:
            raise KeyError("event not found")
        for key, value in values.items():
            setattr(event, key, value)
        self.session.updated[self.event_id] = values
        return 1


def make_zone_snapshot(zone_id="z_1"):
    return ZoneSnapshot(
        zone_id=zone_id,
        camera_id=1,
        name="Test Zone",
        polygon=[[0, 0], [0, 100], [100, 100], [100, 0]],
        color="#FF0000",
        attribution_mode="multiple",
        is_deleted=False,
        properties={},
    )


def _collect_dwell_events(mem_session):
    return [event for event in mem_session.events.values() if hasattr(event, "entry_ts")]


def test_point_in_polygon():
    polygon = [[0, 0], [0, 10], [10, 10], [10, 0]]
    assert point_in_polygon((5, 5), polygon) is True
    assert point_in_polygon((11, 5), polygon) is False


def test_zone_presence_entry_exit_debounce():
    zone_cache = ZoneCache()
    zone_cache._zones_by_camera = {1: [make_zone_snapshot()]}
    mem_session = MemorySession()
    engine = ZonePresenceEngine(session_factory=lambda: mem_session, zone_cache=zone_cache, debounce_ms=10)

    ts = datetime.utcnow()
    detections = [{"bbox": (10, 10, 20, 20), "confidence": 0.9, "class_name": "person"}]
    engine.process_detections(camera_id=1, detections=detections, timestamp=ts)
    engine.process_detections(camera_id=1, detections=detections, timestamp=ts + timedelta(milliseconds=20))

    dwell_events = _collect_dwell_events(mem_session)
    assert len(dwell_events) == 1
    event = dwell_events[0]
    assert event.entry_ts == ts + timedelta(milliseconds=20)

    # Exit after debounce
    engine.process_detections(camera_id=1, detections=[], timestamp=ts + timedelta(milliseconds=40))
    engine.process_detections(camera_id=1, detections=[], timestamp=ts + timedelta(milliseconds=80))

    updated = list(mem_session.updated.values())[0]
    assert "exit_ts" in updated
    assert updated["dwell_seconds"] >= 0
    assert engine.live_snapshot()["z_1"]["count"] == 0


def test_overlapping_zones_create_multiple_events():
    zone_cache = ZoneCache()
    zone_cache._zones_by_camera = {
        1: [
            make_zone_snapshot("z_a"),
            make_zone_snapshot("z_b"),
        ]
    }
    mem_session = MemorySession()
    engine = ZonePresenceEngine(session_factory=lambda: mem_session, zone_cache=zone_cache, debounce_ms=0)

    ts = datetime.utcnow()
    detections = [{"bbox": (10, 10, 20, 20), "confidence": 0.9, "class_name": "person"}]
    engine.process_detections(camera_id=1, detections=detections, timestamp=ts)
    engine.process_detections(camera_id=1, detections=detections, timestamp=ts + timedelta(milliseconds=10))

    assert len(_collect_dwell_events(mem_session)) == 2
    snapshot = engine.live_snapshot()
    assert snapshot["z_a"]["count"] == 1
    assert snapshot["z_b"]["count"] == 1


def test_undo_redo_manager():
    manager = UndoRedoManager(max_depth=5)
    action = ZoneAction(action="create", before=None, after=make_zone_snapshot())
    manager.record(action)
    popped = manager.undo()
    assert popped == action
    redo = manager.redo()
    assert redo == action

