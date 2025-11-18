import numpy as np
import threading
from queue import Queue, Empty
import time
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Literal
import cv2
import face_recognition
from ultralytics import YOLO
from datetime import datetime, timedelta
from collections import defaultdict
import requests
import os
import json
import uuid
import csv
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    Response,
    Body,
    Form,
    status,
    Depends,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
import logging
import subprocess
try:
    from imutils.video import VideoStream
except Exception:
    VideoStream = None
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from threading import Lock
from logging.handlers import RotatingFileHandler
from prometheus_fastapi_instrumentator import Instrumentator

from backend.schemas import (
    ZoneCreate,
    ZoneUpdate,
    ZoneResponse,
    DwellEventResponse,
    DwellTargetResponse,
    DwellTargetUpdate,
    DwellSessionResponse,
    ZoneCounterEventResponse,
)
from .db_settings import init_db, SessionLocal  # Import SessionLocal from db_settings
from .models import (
    DetectionEvent,
    Camera,
    Zone,
    DwellEvent,
    DwellTarget,
    DwellTargetSession,
    ZoneCounterEvent,
)
from backend import models  # Add this import
from .monitoring.metrics import metrics
from .zone_logic import (
    ZoneCache,
    ZonePresenceEngine,
    UndoRedoManager,
    ZoneAction,
    ZoneSnapshot,
    hex_to_bgr,
)
from .services.dwell_tracker import DwellTrackingEngine

# FFmpeg-based RTSP capture fallback when OpenCV VideoCapture can't open the stream
class RTSPFFmpegCapture:
    def __init__(self, url):
        self.url = url
        self.proc = None
        self.stdout = None
        self._buffer = b''
        self.opened = False
        try:
            # Use robust flags for RTSP to reduce frame corruption and force TCP transport
            cmd = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '32',
                '-analyzeduration', '0',
                '-i', self.url,
                '-loglevel', 'error',
                '-an',
                '-r', '10',
                '-f', 'image2pipe',
                '-vcodec', 'mjpeg',
                '-'
            ]
            # Capture stderr so we can log ffmpeg decoder warnings/errors
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.stdout = self.proc.stdout
            # Give ffmpeg a short moment to fail-fast if input can't be opened
            time.sleep(0.4)
            if self.proc.poll() is not None:
                # Process exited; capture stderr for diagnosis and mark not opened
                try:
                    stderr = self.proc.stderr.read().decode('utf-8', errors='replace')
                    app_logger.warning(f"ffmpeg[{self.url}] exited immediately: {stderr.strip()}")
                except Exception:
                    pass
                try:
                    self.proc.kill()
                except Exception:
                    pass
                self.proc = None
                self.stdout = None
                self.opened = False
            else:
                self.opened = True
                if self.proc and self.proc.stderr:
                    t = threading.Thread(target=self._read_stderr, daemon=True)
                    t.start()
        except Exception as e:
            app_logger.error(f"FFmpeg fallback failed to start for {self.url}: {e}")
            self.opened = False

    def _read_stderr(self):
        try:
            for line in iter(self.proc.stderr.readline, b''):
                try:
                    text = line.decode('utf-8', errors='replace').strip()
                    if text:
                        app_logger.warning(f"ffmpeg[{self.url}]: {text}")
                except Exception:
                    pass
        except Exception:
            pass

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened or not self.stdout:
            return False, None

        # Read until we find a JPEG frame (start 0xFFD8, end 0xFFD9)
        try:
            data = self._buffer
            while True:
                chunk = self.stdout.read(4096)
                if not chunk:
                    return False, None
                data += chunk
                start = data.find(b'\xff\xd8')
                end = data.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg = data[start:end+2]
                    self._buffer = data[end+2:]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        return False, None
                    return True, frame
        except Exception as e:
            app_logger.error(f"Error reading ffmpeg pipe for {self.url}: {e}")
            return False, None

    def release(self):
        try:
            if self.proc:
                self.proc.kill()
                self.proc = None
        except Exception:
            pass


# imutils VideoStream wrapper to provide a consistent interface
class ImutilsVideoStreamWrapper:
    def __init__(self, src=0):
        self.src = src
        self.vs = None
        self.opened = False
        self.is_imutils = True
        try:
            if VideoStream is None:
                raise ImportError("imutils not available")
            # VideoStream will start a thread internally; start it immediately
            self.vs = VideoStream(src=src).start()
            # Give it a short moment to warm up
            time.sleep(0.5)
            self.opened = True
        except Exception as e:
            app_logger.warning(f"Imutils VideoStream failed for {src}: {e}")
            self.vs = None
            self.opened = False

    def isOpened(self):
        return bool(self.opened and self.vs is not None)

    def read(self):
        try:
            if not self.vs:
                return False, None
            frame = self.vs.read()
            if frame is None:
                return False, None
            return True, frame
        except Exception:
            return False, None

    def release(self):
        try:
            if self.vs:
                try:
                    # VideoStream provides a stop() method
                    self.vs.stop()
                except Exception:
                    # Fallback to releasing underlying stream
                    try:
                        if hasattr(self.vs, 'stream') and hasattr(self.vs.stream, 'release'):
                            self.vs.stream.release()
                    except Exception:
                        pass
                self.vs = None
        except Exception:
            pass

# Create custom loggers
app_logger = logging.getLogger('app')
fastapi_logger = logging.getLogger('uvicorn.access')
yolo_logger = logging.getLogger('ultralytics')

# Configure app logger
app_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(message)s', 
                            datefmt='%H:%M:%S')

# Console handler with custom formatting
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
app_logger.addHandler(console_handler)

# Suppress other loggers
fastapi_logger.handlers = []
yolo_logger.handlers = []

# Create the FastAPI app
app = FastAPI()

# Add Prometheus instrumentation BEFORE any other setup
Instrumentator().instrument(app).expose(app)

# Allow CORS for React frontend
# Get CORS origins from environment variable
cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
try:
    # Parse the JSON string to get the list of origins
    allowed_origins = json.loads(cors_origins)
except json.JSONDecodeError:
    # Fallback to allowing all origins if parsing fails
    allowed_origins = ["*"]
    app_logger.warning("Failed to parse CORS_ORIGINS, falling back to allow all origins")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
video_captures = {}  # Dictionary to store VideoCapture objects for each camera
latest_frames = {}   # Dictionary to store latest frames for each camera
active_streams = {}
fps_stats = {}
sources = []
cameras = []
recent_alerts = []
alert_queue = Queue()
model = None # Load the YOLO model

# Global variable for webcam control
cap = None
latest_frame = None
running = False

# Replace the global detection_times with a nested dictionary
detection_times = defaultdict(lambda: defaultdict(lambda: datetime.min))

# Replace the global variable with a dictionary to track per-camera detections
detected_objects_this_session = {}  # Changed from set() to dict()

# Add these at the top with other global variables
camera_threads = {}  # Store threads for each camera
camera_running = {}  # Track running state for each camera
frame_locks = {}    # Locks for thread-safe frame access

zone_cache = ZoneCache()
undo_manager = UndoRedoManager(max_depth=50)
zone_presence_engine = ZonePresenceEngine(SessionLocal, zone_cache)
dwell_tracker = DwellTrackingEngine(SessionLocal, zone_cache)
TARGET_IMAGE_DIR = Path(__file__).resolve().parent / "storage" / "dwell_targets"
TARGET_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_cameras():
    """Fetch all cameras from the API"""
    try:
        response = requests.get("http://localhost:8000/api/cameras")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch cameras: {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server")
        return []
    except Exception as e:
        print(f"Error fetching cameras: {str(e)}")
        return []

def parse_zone_ids(payload: Optional[str]) -> List[str]:
    if payload is None:
        return []
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item]
    except json.JSONDecodeError:
        pass
    if isinstance(payload, str):
        return [segment.strip() for segment in payload.split(",") if segment.strip()]
    return []


def ensure_uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}") from exc


def zone_to_response(zone: Zone) -> ZoneResponse:
    return ZoneResponse(
        zone_id=zone.zone_id,
        camera_id=zone.camera_id,
        name=zone.name,
        polygon=zone.polygon,
        color=zone.color,
        attribution_mode=zone.attribution_mode,
        properties=zone.properties or {},
        is_deleted=zone.is_deleted,
        created_at=zone.created_at,
        updated_at=zone.updated_at,
    )


def apply_zone_snapshot(session: Session, snapshot: ZoneSnapshot) -> Zone:
    zone = session.query(Zone).filter(Zone.zone_id == snapshot.zone_id).first()
    if zone is None:
        zone = Zone(
            zone_id=snapshot.zone_id,
            camera_id=snapshot.camera_id,
            name=snapshot.name,
            polygon=snapshot.polygon,
            color=snapshot.color,
            attribution_mode=snapshot.attribution_mode,
            is_deleted=snapshot.is_deleted,
            properties=snapshot.properties or {},
        )
        session.add(zone)
    else:
        zone.camera_id = snapshot.camera_id
        zone.name = snapshot.name
        zone.polygon = snapshot.polygon
        zone.color = snapshot.color
        zone.attribution_mode = snapshot.attribution_mode
        zone.is_deleted = snapshot.is_deleted
        zone.properties = snapshot.properties or {}
    return zone


def delete_zone_record(session: Session, zone_id: str, purge: bool = False) -> bool:
    zone = session.query(Zone).filter(Zone.zone_id == zone_id).first()
    if not zone:
        return False
    if purge:
        session.delete(zone)
    else:
        zone.is_deleted = True
    return True


def process_frame(frame, result, camera_id):
    """Process a single frame and apply detections based on model type."""
    with metrics.measure_latency(str(camera_id), model.task):
        try:
            annotated_frame = frame.copy()
            alert = None
            zone_detections = []
            # Initialize camera's detection set if not exists
            if camera_id not in detected_objects_this_session:
                detected_objects_this_session[camera_id] = set()

            # Define color constants
            KEYPOINT_COLOR = (0, 255, 0)  # Green
            SKELETON_COLOR = (0, 255, 255)  # Yellow
            DETECTION_COLORS = {
                "person": (0, 0, 255),    # Red
                "bottle": (0, 255, 0),    # Green
                "potted plant": (255, 0, 0)  # Blue
            }

            # Initialize variables for metrics recording
            # These will only be updated if a valid detection is found
            valid_detection = False
            recorded_class_name = "unknown"
            recorded_confidence = 0.0

            try:
                # Handle pose estimation
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    for person in result.keypoints:
                        keypoints = person.data[0].cpu().numpy()
                        # Draw keypoints
                        for kp in keypoints:
                            if kp[2] > 0.5:  # Confidence threshold
                                x, y = int(kp[0]), int(kp[1])
                                cv2.circle(annotated_frame, (x, y), 4, KEYPOINT_COLOR, -1)
                        # Define skeleton connections (COCO format)
                        skeleton = [
                            [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13],
                            [6,7], [6,8], [7,9], [8,10], [9,11], [2,3], [1,2], [1,3],
                            [2,4], [3,5], [4,6], [5,7]
                        ]
                        # Draw skeleton
                        for connection in skeleton:
                            start_idx = connection[0] - 1
                            end_idx = connection[1] - 1
                            if (keypoints[start_idx][2] > 0.5 and
                                keypoints[end_idx][2] > 0.5):
                                start_point = (int(keypoints[start_idx][0]),
                                     int(keypoints[start_idx][1]))
                                end_point = (int(keypoints[end_idx][0]),
                                   int(keypoints[end_idx][1]))
                                cv2.line(annotated_frame, start_point, end_point,
                                        SKELETON_COLOR, 2)

                # Handle segmentation
                if hasattr(result, 'masks') and result.masks is not None:
                    for i, mask in enumerate(result.masks):
                        try:
                            class_id = int(result.boxes[i].cls[0])
                            class_name = result.names[class_id]
                            if class_name in DETECTION_COLORS:
                                mask_array = mask.data.cpu().numpy()[0]
                                # Resize mask if needed
                                if mask_array.shape[:2] != frame.shape[:2]:
                                    mask_array = cv2.resize(
                                        mask_array,
                                        (frame.shape[1], frame.shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                    )
                                # Create and apply mask overlay
                                binary_mask = (mask_array > 0.5).astype(np.uint8)
                                color = DETECTION_COLORS[class_name]
                                mask_overlay = np.zeros_like(frame)
                                mask_overlay[binary_mask == 1] = color
                                # Blend with original frame
                                mask_area = (binary_mask > 0)
                                annotated_frame[mask_area] = cv2.addWeighted(
                                    annotated_frame[mask_area],
                                    0.6,  # Original frame weight
                                    mask_overlay[mask_area],
                                    0.4,  # Mask weight
                                    0
                                )
                        except Exception as e:
                            logging.error(f"Error processing mask {i}: {str(e)}")
                            continue

                # Handle object detection boxes
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        try:
                            # Get box information
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = result.names[class_id]
                            if class_name in DETECTION_COLORS:
                                color = DETECTION_COLORS[class_name]
                                # Draw bounding box
                                cv2.rectangle(
                                    annotated_frame,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    color,
                                    2
                                )
                                # Add label
                                label = f"{class_name} {confidence:.2f}"
                                label_size, baseline = cv2.getTextSize(
                                    label,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    2
                                )
                                # Draw label background
                                label_y = max(y1 - 10, label_size[1])
                                cv2.rectangle(
                                    annotated_frame,
                                    (int(x1), int(label_y - label_size[1])),
                                    (int(x1 + label_size[0]), int(label_y + baseline)),
                                    color,
                                    cv2.FILLED
                                )
                                # Draw label text
                                cv2.putText(
                                    annotated_frame,
                                    label,
                                    (int(x1), int(label_y)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    2
                                )

                                # Check if this specific box is a valid detection for our criteria
                                is_valid_for_box = (
                                    class_name == "person" or
                                    (class_name == "bottle" and model.task == "detect")
                                )

                                # If it's a valid detection, update the overall flag and record details
                                zone_detections.append(
                                    {
                                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                                        "confidence": confidence,
                                        "class_name": class_name,
                                    }
                                )

                                if is_valid_for_box:
                                    valid_detection = True
                                    recorded_class_name = class_name
                                    recorded_confidence = confidence

                                    # Apply the 10-second cooldown check
                                    current_time = datetime.now()
                                    if detection_times[camera_id][class_name] == datetime.min:
                                        detection_times[camera_id][class_name] = current_time
                                    elif current_time - detection_times[camera_id][class_name] >= timedelta(seconds=10):
                                        model_type = "objectDetection" if model.task == "detect" else (
                                            "segmentation" if model.task == "segment" else "pose"
                                        )
                                        save_event_to_db(class_name, model_type, camera_id)
                                        detected_objects_this_session[camera_id].add(class_name)
                                        # Reset the timer for this class
                                        detection_times[camera_id][class_name] = current_time

                        except Exception as e:
                            logging.error(f"Error processing box: {str(e)}")
                            continue

                # Add frame metadata
                cv2.putText(
                    annotated_frame,
                    f"Frame Size: {frame.shape[1]}x{frame.shape[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # Record metrics only if a valid detection was found in this frame
                if valid_detection:
                    metrics.record_detection(
                        camera_id=str(camera_id),
                        class_name=recorded_class_name,
                        model_type=model.task,
                        confidence=recorded_confidence
                    )

            except Exception as e:
                logging.error(f"Error in process_frame inner try: {str(e)}")
                return frame, None  # Return original frame on error

            try:
                zone_presence_engine.process_detections(
                    camera_id=camera_id,
                    detections=zone_detections,
                    timestamp=datetime.utcnow(),
                )
                zones = zone_cache.get_for_camera(camera_id)
                live_snapshot = zone_presence_engine.live_snapshot()
                for zone in zones:
                    polygon = np.array(zone.polygon, dtype=np.int32)
                    if polygon.ndim == 2:
                        polygon_to_draw = polygon.reshape((-1, 1, 2))
                    else:
                        polygon_to_draw = polygon
                    color = hex_to_bgr(zone.color or "#FF0000")
                    cv2.polylines(annotated_frame, [polygon_to_draw], isClosed=True, color=color, thickness=2)
                    zone_label = f"{zone.name} ({live_snapshot.get(zone.zone_id, {}).get('count', 0)})"
                    if polygon.shape[0] > 0:
                        label_point = tuple(polygon[0])
                        cv2.putText(
                            annotated_frame,
                            zone_label,
                            (int(label_point[0]), int(label_point[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
            except Exception as zone_err:
                logging.error(f"Zone overlay error for camera {camera_id}: {zone_err}")

            try:
                dwell_tracker.process_frame(camera_id=camera_id, frame=frame, timestamp=datetime.utcnow())
            except Exception as dwell_err:
                logging.error(f"Dwell tracker error for camera {camera_id}: {dwell_err}")

            return annotated_frame, alert
        except Exception as e:
            metrics.record_error(
                camera_id=str(camera_id),
                error_type=type(e).__name__,
                component="frame_processing"
            )
            logging.error(f"Error in process_frame outer try: {str(e)}")
            return frame, None
def capture_frames_for_camera(camera_id):
    """Capture and process frames for a single camera"""
    # Get camera name from database
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_name = camera.source_name if camera else f"Camera {camera_id}"
    finally:
        db.close()

    current_thread = threading.current_thread()
    current_thread.name = f"🎥 {camera_name}"  # Use actual camera name
    
    thread_info = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "thread_name": current_thread.name,
        "thread_id": current_thread.ident,
        "start_time": datetime.now().strftime("%H:%M:%S")
    }
    
    # Use custom logger with camera name
    app_logger.info("╔════════════════════════════════════")
    app_logger.info(f"║ {camera_name} Thread Started")
    app_logger.info(f"║ Thread Name: {current_thread.name}")
    app_logger.info(f"║ Thread ID: {current_thread.ident}")
    app_logger.info("╚════════════════════════════════════")
    
    if not hasattr(app, 'camera_threads_info'):
        app.camera_threads_info = {}
    app.camera_threads_info[camera_id] = thread_info
    
    try:
        cap = video_captures[camera_id]
        while camera_running.get(camera_id, False):
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame from camera {camera_id}")
                # Attempt a single reopen for RTSP cameras to recover transient issues
                try:
                    db_retry = SessionLocal()
                    cam_retry = db_retry.query(Camera).filter(Camera.id == camera_id).first()
                finally:
                    db_retry.close()

                if cam_retry and (cam_retry.stream_type or '').lower() != 'live':
                    app_logger.info(f"Attempting to reopen RTSP for camera {camera_id}")
                    try:
                        try:
                            cap.release()
                        except Exception:
                            pass
                        # Try imutils threaded VideoStream first
                        try:
                            new_cap = ImutilsVideoStreamWrapper(cam_retry.stream)
                        except Exception:
                            new_cap = None

                        if new_cap and new_cap.isOpened():
                            video_captures[camera_id] = new_cap
                            cap = new_cap
                            app_logger.info(f"Reopened RTSP stream via imutils for camera {camera_id}")
                            continue
                        else:
                            # Try ffmpeg fallback
                            ff = RTSPFFmpegCapture(cam_retry.stream)
                            if ff.isOpened():
                                video_captures[camera_id] = ff
                                cap = ff
                                app_logger.info(f"Reopened RTSP stream via ffmpeg for camera {camera_id}")
                                continue
                            app_logger.error(f"Reopen failed for camera {camera_id}. Will retry later.")
                    except Exception as e:
                        app_logger.error(f"Error reopening RTSP for camera {camera_id}: {e}")

                # Avoid tight loop on read failures
                time.sleep(0.5)
                continue

            try:
                results = model(frame)
                annotated_frame, alert = process_frame(frame, results[0], camera_id)
                
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                with frame_locks[camera_id]:
                    latest_frames[camera_id] = buffer.tobytes()
            except Exception as e:
                logging.error(f"Error processing frame for camera {camera_id}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Error in capture_frames for {camera_name}: {str(e)}")
    finally:
        if hasattr(app, 'camera_threads_info') and camera_id in app.camera_threads_info:
            del app.camera_threads_info[camera_id]

@app.get("/api/cameras")
def get_cameras():
    db = SessionLocal()
    try:
        cameras = db.query(Camera).all()
        camera_list = []
        for camera in cameras:
            camera_list.append({
                "id": camera.id,
                "source_name": camera.source_name,
                "stream_type": camera.stream_type,
                "stream": camera.stream,
                "location": camera.location,
                "created_at": camera.created_at.strftime("%Y-%m-%d %H:%M:%S") if camera.created_at else None
            })
        return camera_list
    except Exception as e:
        print(f"Error fetching cameras: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching cameras: {str(e)}"
        )
    finally:
        db.close()

@app.post("/api/cameras")
async def add_camera(camera: dict):
    """Create a new camera entry - redirects to create_camera endpoint"""
    return await create_camera(camera)

@app.get("/api/alerts")
def get_alerts():
    """Fetch recent alerts"""
    return recent_alerts


# ----------------------------
# Zone management & analytics
# ----------------------------


@app.get("/zones", response_model=List[ZoneResponse])
def list_zones(camera_id: Optional[int] = Query(default=None), include_deleted: bool = Query(default=False)):
    session = SessionLocal()
    try:
        query = session.query(Zone)
        if camera_id is not None:
            query = query.filter(Zone.camera_id == camera_id)
        if not include_deleted:
            query = query.filter(Zone.is_deleted.is_(False))
        zones = query.order_by(Zone.created_at.asc()).all()
        return [zone_to_response(zone) for zone in zones]
    finally:
        session.close()


@app.post("/zones", response_model=ZoneResponse, status_code=status.HTTP_201_CREATED)
def create_zone(zone_input: ZoneCreate):
    session = SessionLocal()
    try:
        zone_id = zone_input.zone_id or f"z_{uuid.uuid4().hex[:8]}"
        new_zone = Zone(
            zone_id=zone_id,
            camera_id=zone_input.camera_id,
            name=zone_input.name,
            polygon=zone_input.polygon,
            color=zone_input.color,
            attribution_mode=zone_input.attribution_mode,
            properties=zone_input.properties or {},
        )
        session.add(new_zone)
        session.commit()
        session.refresh(new_zone)

        snapshot = ZoneSnapshot.from_model(new_zone)
        undo_manager.record(ZoneAction(action="create", before=None, after=snapshot))
        zone_cache.refresh(SessionLocal)
        return zone_to_response(new_zone)
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=400, detail=f"Unable to create zone: {exc.orig}")
    finally:
        session.close()


@app.put("/zones/{zone_id}", response_model=ZoneResponse)
def update_zone(zone_id: str, updates: ZoneUpdate):
    session = SessionLocal()
    try:
        zone = session.query(Zone).filter(Zone.zone_id == zone_id).first()
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")

        before = ZoneSnapshot.from_model(zone)
        data = updates.model_dump(exclude_unset=True)
        for field, value in data.items():
            setattr(zone, field, value)
        session.commit()
        session.refresh(zone)

        undo_manager.record(ZoneAction(action="update", before=before, after=ZoneSnapshot.from_model(zone)))
        zone_cache.refresh(SessionLocal)
        return zone_to_response(zone)
    finally:
        session.close()


@app.delete("/zones/{zone_id}")
def delete_zone(zone_id: str, purge: bool = Query(default=False)):
    session = SessionLocal()
    try:
        zone = session.query(Zone).filter(Zone.zone_id == zone_id).first()
        if not zone:
            raise HTTPException(status_code=404, detail="Zone not found")

        before = ZoneSnapshot.from_model(zone)
        delete_zone_record(session, zone_id, purge=purge)
        session.commit()

        undo_manager.record(
            ZoneAction(
                action="delete",
                before=before,
                after=None,
                metadata={"purge": purge},
            )
        )
        zone_cache.refresh(SessionLocal)
        zone_presence_engine.drop_zone(zone_id)
        return {"status": "deleted", "zone_id": zone_id, "purge": purge}
    finally:
        session.close()


def _apply_zone_action(action: ZoneAction, inverse: bool = False) -> ZoneResponse:
    session = SessionLocal()
    try:
        target_snapshot = None
        purge = action.metadata.get("purge", False)
        if action.action == "create":
            if inverse:
                delete_zone_record(session, action.after.zone_id, purge=True)
                zone_presence_engine.drop_zone(action.after.zone_id)
                response = None
            else:
                target_snapshot = action.after
        elif action.action == "delete":
            if inverse:
                target_snapshot = action.before
            else:
                delete_zone_record(session, action.before.zone_id, purge=purge)
                zone_presence_engine.drop_zone(action.before.zone_id)
                response = None
        elif action.action == "update":
            target_snapshot = action.before if inverse else action.after
        else:
            raise HTTPException(status_code=400, detail="Unknown action")

        if target_snapshot:
            zone = apply_zone_snapshot(session, target_snapshot)
            session.flush()
            response = zone_to_response(zone)

        session.commit()
        zone_cache.refresh(SessionLocal)
        return response
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@app.post("/zones/undo")
def undo_zone_change():
    action = undo_manager.undo()
    if not action:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    response = _apply_zone_action(action, inverse=True)
    return {"status": "ok", "action": action.action, "zone": response}


@app.post("/zones/redo")
def redo_zone_change():
    action = undo_manager.redo()
    if not action:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    response = _apply_zone_action(action, inverse=False)
    return {"status": "ok", "action": action.action, "zone": response}


@app.get("/events", response_model=List[DwellEventResponse])
def list_dwell_events(
    zone_id: Optional[str] = Query(default=None),
    object_id: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    limit: int = Query(default=200, le=1000),
):
    session = SessionLocal()
    try:
        query = session.query(DwellEvent).order_by(DwellEvent.entry_ts.desc())
        if zone_id:
            query = query.filter(DwellEvent.zone_id == zone_id)
        if object_id:
            query = query.filter(DwellEvent.object_id == object_id)
        if start:
            query = query.filter(DwellEvent.entry_ts >= start)
        if end:
            query = query.filter(DwellEvent.entry_ts <= end)

        events = query.limit(limit).all()
        return [DwellEventResponse.model_validate(event) for event in events]
    finally:
        session.close()

# ----------------------------
# Dwell tracking module
# ----------------------------


@app.post("/dwell/targets", response_model=DwellTargetResponse, status_code=status.HTTP_201_CREATED)
async def create_dwell_target(
    name: str = Form(...),
    camera_id: int = Form(...),
    zone_ids: str = Form(...),
    match_threshold: float = Form(0.45),
    face_image: UploadFile = File(...),
):
    """Create a new dwell target with face recognition."""
    try:
        # Validate camera exists
        db = SessionLocal()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        db.close()
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with id {camera_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error checking camera: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating camera: {e}")

    try:
        zones = parse_zone_ids(zone_ids)
        if not zones:
            raise HTTPException(status_code=400, detail="At least one zone_id is required")
    except Exception as e:
        app_logger.error(f"Error parsing zone_ids: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid zone_ids format: {e}")

    try:
        file_bytes = await face_image.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Face image is required")
    except Exception as e:
        app_logger.error(f"Error reading face image: {e}")
        raise HTTPException(status_code=400, detail=f"Error reading image file: {e}")

    try:
        np_image = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data - could not decode image")
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        app_logger.error(f"Error converting image color space: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image colors: {e}")

    try:
        # Check if face_recognition is available
        try:
            encodings = face_recognition.face_encodings(rgb_image)
        except Exception as e:
            app_logger.error(f"face_recognition library error: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Face recognition library error. Please ensure face_recognition and dlib are properly installed: {e}"
            )
        
        if not encodings:
            raise HTTPException(
                status_code=400, 
                detail="No face detected in supplied image. Please upload an image with a clear, visible face."
            )
        
        if len(encodings) > 1:
            app_logger.warning(f"Multiple faces detected in image, using the first one")
        
        encoding_list = encodings[0].tolist()
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error in face encoding: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing face: {e}")

    try:
        filename = f"{uuid.uuid4().hex}_{face_image.filename or 'target.jpg'}"
        target_path = TARGET_IMAGE_DIR / filename
        with open(target_path, "wb") as file_obj:
            file_obj.write(file_bytes)
    except Exception as e:
        app_logger.error(f"Error saving target image: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving image file: {e}")

    session = SessionLocal()
    try:
        record = DwellTarget(
            name=name.strip(),
            camera_id=camera_id,
            zone_ids=zones,
            face_encoding=encoding_list,
            match_threshold=match_threshold,
            reference_image_path=str(target_path),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        dwell_tracker.invalidate_cache()
        app_logger.info(f"Successfully created dwell target: {record.target_id} for camera {camera_id}")
        return DwellTargetResponse.model_validate(record)
    except IntegrityError as e:
        session.rollback()
        app_logger.error(f"Database integrity error creating target: {e}")
        raise HTTPException(status_code=400, detail=f"Database error: {e}")
    except Exception as exc:
        session.rollback()
        app_logger.error(f"Error creating dwell target: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unable to create target: {exc}") from exc
    finally:
        session.close()


@app.get("/dwell/targets", response_model=List[DwellTargetResponse])
def list_dwell_targets():
    session = SessionLocal()
    try:
        records = session.query(DwellTarget).order_by(DwellTarget.created_at.desc()).all()
        return [DwellTargetResponse.model_validate(item) for item in records]
    finally:
        session.close()


@app.put("/dwell/targets/{target_id}", response_model=DwellTargetResponse)
def update_dwell_target(target_id: str, payload: DwellTargetUpdate):
    session = SessionLocal()
    try:
        target_uuid = ensure_uuid(target_id, "target_id")
        target = session.query(DwellTarget).filter(DwellTarget.target_id == target_uuid).first()
        if not target:
            raise HTTPException(status_code=404, detail="Target not found")

        data = payload.model_dump(exclude_unset=True)
        if "zone_ids" in data and data["zone_ids"] is not None and len(data["zone_ids"]) == 0:
            raise HTTPException(status_code=400, detail="zone_ids cannot be empty")

        for field, value in data.items():
            setattr(target, field, value)
        session.commit()
        session.refresh(target)
        dwell_tracker.invalidate_cache()
        return DwellTargetResponse.model_validate(target)
    finally:
        session.close()


@app.delete("/dwell/targets/{target_id}")
def delete_dwell_target(target_id: str):
    session = SessionLocal()
    try:
        target_uuid = ensure_uuid(target_id, "target_id")
        target = session.query(DwellTarget).filter(DwellTarget.target_id == target_uuid).first()
        if not target:
            raise HTTPException(status_code=404, detail="Target not found")
        image_path = target.reference_image_path
        session.delete(target)
        session.commit()
        dwell_tracker.invalidate_cache()
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass
        return {"status": "deleted", "target_id": target_id}
    finally:
        session.close()


@app.get("/dwell/sessions", response_model=List[DwellSessionResponse])
def list_dwell_sessions(
    target_id: Optional[str] = Query(default=None),
    zone_id: Optional[str] = Query(default=None),
    limit: int = Query(default=200, le=1000),
):
    session = SessionLocal()
    try:
        query = session.query(DwellTargetSession).order_by(DwellTargetSession.entry_ts.desc())
        if target_id:
            target_uuid = ensure_uuid(target_id, "target_id")
            query = query.filter(DwellTargetSession.target_id == target_uuid)
        if zone_id:
            query = query.filter(DwellTargetSession.zone_id == zone_id)
        records = query.limit(limit).all()
        return [DwellSessionResponse.model_validate(item) for item in records]
    finally:
        session.close()


@app.get("/dwell/live")
def dwell_live_sessions():
    live = dwell_tracker.live_sessions()
    if not live:
        return {"sessions": []}

    session = SessionLocal()
    try:
        target_ids = []
        for entry in live:
            try:
                target_ids.append(uuid.UUID(entry["target_id"]))
            except Exception:
                continue
        targets = []
        if target_ids:
            targets = session.query(DwellTarget).filter(DwellTarget.target_id.in_(target_ids)).all()
        target_map = {str(item.target_id): item for item in targets}

        zone_ids = {entry["zone_id"] for entry in live if entry.get("zone_id")}
        zones = []
        if zone_ids:
            zones = session.query(Zone).filter(Zone.zone_id.in_(zone_ids)).all()
        zone_map = {zone.zone_id: zone for zone in zones}

        enriched = []
        for entry in live:
            target = target_map.get(entry["target_id"])
            zone = zone_map.get(entry["zone_id"])
            enriched.append(
                {
                    **entry,
                    "target_name": target.name if target else None,
                    "zone_name": zone.name if zone else None,
                }
            )
        return {"sessions": enriched}
    finally:
        session.close()


# ----------------------------
# Zone counter endpoints
# ----------------------------


@app.get("/zone-counters/live")
def zone_counters_live(
    camera_id: Optional[int] = Query(default=None),
    zone_id: Optional[str] = Query(default=None),
    object_type: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
):
    session = SessionLocal()
    try:
        aggregates_query = session.query(
            ZoneCounterEvent.zone_id,
            ZoneCounterEvent.object_type,
            ZoneCounterEvent.event_type,
            func.count(ZoneCounterEvent.event_id).label("count"),
        )
        if camera_id:
            aggregates_query = aggregates_query.filter(ZoneCounterEvent.camera_id == camera_id)
        if zone_id:
            aggregates_query = aggregates_query.filter(ZoneCounterEvent.zone_id == zone_id)
        if object_type:
            aggregates_query = aggregates_query.filter(ZoneCounterEvent.object_type == object_type)
        if start:
            aggregates_query = aggregates_query.filter(ZoneCounterEvent.timestamp >= start)
        if end:
            aggregates_query = aggregates_query.filter(ZoneCounterEvent.timestamp <= end)

        aggregates = aggregates_query.group_by(
            ZoneCounterEvent.zone_id,
            ZoneCounterEvent.object_type,
            ZoneCounterEvent.event_type,
        ).all()

        aggregate_map: Dict[str, Dict[str, Dict[str, int]]] = {}
        for agg_zone_id, agg_object, agg_type, agg_count in aggregates:
            zone_bucket = aggregate_map.setdefault(agg_zone_id, {})
            type_bucket = zone_bucket.setdefault(agg_object, {"enter": 0, "exit": 0})
            type_bucket[agg_type] = agg_count

        live_snapshot = zone_presence_engine.live_snapshot()
        zones_query = session.query(Zone).filter(Zone.is_deleted.is_(False))
        if camera_id:
            zones_query = zones_query.filter(Zone.camera_id == camera_id)
        if zone_id:
            zones_query = zones_query.filter(Zone.zone_id == zone_id)

        response_payload = []
        zones = zones_query.order_by(Zone.created_at.asc()).all()
        for zone in zones:
            zone_live = live_snapshot.get(zone.zone_id, {"count": 0, "objects": [], "by_type": {}})
            totals = {}
            stats = aggregate_map.get(zone.zone_id, {})
            by_type = zone_live.get("by_type", {})
            for obj_type, counts in stats.items():
                totals[obj_type] = {
                    "entered": counts.get("enter", 0),
                    "exited": counts.get("exit", 0),
                    "current": by_type.get(obj_type, {}).get("count", 0),
                }
            response_payload.append(
                {
                    "zone": zone_to_response(zone),
                    "live": zone_live,
                    "totals": totals,
                }
            )
        return response_payload
    finally:
        session.close()


@app.get("/zone-counters/events", response_model=List[ZoneCounterEventResponse])
def list_zone_counter_events(
    camera_id: Optional[int] = Query(default=None),
    zone_id: Optional[str] = Query(default=None),
    object_type: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    limit: int = Query(default=500, le=2000),
):
    session = SessionLocal()
    try:
        query = session.query(ZoneCounterEvent).order_by(ZoneCounterEvent.timestamp.desc())
        if camera_id:
            query = query.filter(ZoneCounterEvent.camera_id == camera_id)
        if zone_id:
            query = query.filter(ZoneCounterEvent.zone_id == zone_id)
        if object_type:
            query = query.filter(ZoneCounterEvent.object_type == object_type)
        if start:
            query = query.filter(ZoneCounterEvent.timestamp >= start)
        if end:
            query = query.filter(ZoneCounterEvent.timestamp <= end)
        records = query.limit(limit).all()
        return [ZoneCounterEventResponse.model_validate(event) for event in records]
    finally:
        session.close()


@app.get("/zone-counters/export")
def export_zone_counter_events(
    camera_id: Optional[int] = Query(default=None),
    zone_id: Optional[str] = Query(default=None),
    object_type: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    format: Literal["csv", "json"] = Query(default="csv"),
):
    session = SessionLocal()
    try:
        query = session.query(ZoneCounterEvent).order_by(ZoneCounterEvent.timestamp.desc())
        if camera_id:
            query = query.filter(ZoneCounterEvent.camera_id == camera_id)
        if zone_id:
            query = query.filter(ZoneCounterEvent.zone_id == zone_id)
        if object_type:
            query = query.filter(ZoneCounterEvent.object_type == object_type)
        if start:
            query = query.filter(ZoneCounterEvent.timestamp >= start)
        if end:
            query = query.filter(ZoneCounterEvent.timestamp <= end)
        records = query.all()

        if format == "json":
            payload = [ZoneCounterEventResponse.model_validate(event).model_dump() for event in records]
            return JSONResponse(content=payload)

        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=["event_id", "timestamp", "zone_id", "camera_id", "object_id", "object_type", "event_type"],
        )
        writer.writeheader()
        for event in records:
            writer.writerow(
                {
                    "event_id": str(event.event_id),
                    "timestamp": event.timestamp.isoformat(),
                    "zone_id": event.zone_id,
                    "camera_id": event.camera_id,
                    "object_id": event.object_id,
                    "object_type": event.object_type,
                    "event_type": event.event_type,
                }
            )
        buffer.seek(0)
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="zone-events.csv"'},
        )
    finally:
        session.close()


@app.get("/zones/{zone_id}/stats")
def zone_stats(
    zone_id: str,
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
):
    session = SessionLocal()
    try:
        query = session.query(DwellEvent).filter(DwellEvent.zone_id == zone_id)
        if start:
            query = query.filter(DwellEvent.entry_ts >= start)
        if end:
            query = query.filter(DwellEvent.entry_ts <= end)

        total_presence = query.with_entities(func.coalesce(func.sum(DwellEvent.dwell_seconds), 0)).scalar() or 0
        entries_count = query.count()
        average_dwell = query.with_entities(func.avg(DwellEvent.dwell_seconds)).scalar() or 0

        live_snapshot = zone_presence_engine.live_snapshot().get(zone_id, {"count": 0, "objects": []})
        return {
            "zone_id": zone_id,
            "start_ts": start.isoformat() if start else None,
            "end_ts": end.isoformat() if end else None,
            "total_presence_seconds": float(total_presence or 0),
            "entries_count": entries_count,
            "average_dwell_seconds": float(average_dwell or 0),
            "live_count": live_snapshot["count"],
            "live_objects": live_snapshot["objects"],
        }
    finally:
        session.close()


@app.get("/zones/live")
def live_zones():
    live_snapshot = zone_presence_engine.live_snapshot()
    session = SessionLocal()
    try:
        zones = session.query(Zone).filter(Zone.is_deleted.is_(False)).all()
        response = []
        for zone in zones:
            live_info = live_snapshot.get(zone.zone_id, {"count": 0, "objects": []})
            response.append(
                {
                    "zone": zone_to_response(zone),
                    "live": live_info,
                }
            )
        return response
    finally:
        session.close()


class ModelRequest(BaseModel):
    model_type: str

@app.post("/start_webcam_stream")
async def start_webcam_stream(request: ModelRequest):
    """Start webcam streams with separate thread per camera."""
    global model, detected_objects_this_session
    
    try:
        # Set up model based on request
        if request.model_type == 'objectDetection':
            model = YOLO('yolov8m.pt')
        elif request.model_type == 'segmentation':
            model = YOLO('yolov8m-seg.pt')
        elif request.model_type == 'pose':
            model = YOLO('yolov8m-pose.pt')
        else:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        # Get all cameras and include both local 'live' and RTSP cameras
        db = SessionLocal()
        all_cameras = db.query(Camera).all()
        # Filter in Python to be resilient to casing or variations
        live_cameras = [cam for cam in all_cameras if (cam.stream_type or '').lower() in ('live', 'rtsp')]
        db.close()

        detected_objects_this_session.clear()

        # Start a thread for each camera
        app_logger.info("╔═══════════════════════════════")
        app_logger.info("║ Starting Camera Streams")
        app_logger.info("╚═══════════════════════════════")
        
        for camera in live_cameras:
            if camera.id not in camera_threads or not camera_threads[camera.id].is_alive():
                # Initialize camera resources
                # Use local webcam device for 'live' type, otherwise use camera.stream for RTSP
                try:
                    # Prefer imutils threaded VideoStream for better FPS/latency
                    try:
                        if (camera.stream_type or '').lower() == 'live':
                            app_logger.info(f"Attempting to open local webcam via imutils for camera {camera.id}")
                            cap_obj = ImutilsVideoStreamWrapper(0)
                        else:
                            app_logger.info(f"Attempting to open RTSP via imutils for camera {camera.id}. URL: {getattr(camera, 'stream', None)}")
                            cap_obj = ImutilsVideoStreamWrapper(camera.stream)
                    except Exception as e:
                        app_logger.warning(f"Imutils open failed for camera {camera.id}: {e}")
                        cap_obj = None

                    if not cap_obj or not cap_obj.isOpened():
                        # Try FFmpeg fallback for RTSP streams
                        if (camera.stream_type or '').lower() != 'live':
                            ff = RTSPFFmpegCapture(camera.stream)
                            if ff.isOpened():
                                video_captures[camera.id] = ff
                                app_logger.info(f"Camera {camera.id} opened via FFmpeg fallback (proc started)")
                            else:
                                app_logger.error(f"FFmpeg fallback not opened for camera {camera.id}. proc stdout: {bool(getattr(ff, 'stdout', None))}")
                                app_logger.error(f"Failed to open capture for camera {camera.id}. URL: {getattr(camera, 'stream', None)}")
                                continue
                        else:
                            app_logger.error(f"Failed to open capture for camera {camera.id}. URL: {getattr(camera, 'stream', None)}")
                            continue
                    else:
                        video_captures[camera.id] = cap_obj
                        backend_name = 'imutils' if getattr(cap_obj, 'isOpened', None) and hasattr(cap_obj, 'vs') else 'imutils'
                        app_logger.info(f"Camera {camera.id} opened via {backend_name} (isOpened={cap_obj.isOpened()})")
                except Exception as e:
                    app_logger.error(f"Failed to open capture for camera {camera.id}: {e}")
                    continue
                frame_locks[camera.id] = Lock()
                camera_running[camera.id] = True
                detected_objects_this_session[camera.id] = set()
                
                # Create and start thread for this camera
                thread = threading.Thread(
                    target=capture_frames_for_camera,
                    args=(camera.id,),
                    daemon=True
                )
                thread.start()
                camera_threads[camera.id] = thread
                
                app_logger.info(f"► Initializing Camera {camera.id}")
                
        return {"message": "Camera streams started", "model_type": request.model_type}
    
    except Exception as e:
        app_logger.error(f"Error starting streams: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_webcam_stream")
async def stop_webcam_stream():
    """Stop all camera streams."""
    try:
        app_logger.info("╔═══════════════════════════════")
        app_logger.info("║ 🛑 Stopping Camera Streams")
        app_logger.info("╚═══════════════════════════════")
        
        # Signal all threads to stop
        for camera_id in list(camera_running.keys()):
            camera_running[camera_id] = False
            
            # Wait for thread to finish
            if camera_id in camera_threads:
                thread = camera_threads[camera_id]
                thread.join(timeout=2.0)
                
                if not thread.is_alive():
                    app_logger.info(f"🎥 Camera {camera_id} - Thread stopped")
                    
                    # Clean up resources
                    if camera_id in video_captures:
                        video_captures[camera_id].release()
                        del video_captures[camera_id]
                        del frame_locks[camera_id]
                        del latest_frames[camera_id]
                        del camera_threads[camera_id]
                        del camera_running[camera_id]

        return {"message": "Camera streams stopped"}
    
    except Exception as e:
        app_logger.error(f"💥 Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/process_frame/{camera_id}")
async def process_frame_endpoint(camera_id: int):
    """Return the latest processed frame for a specific camera."""
    # Check if camera stream is running
    if camera_id not in camera_running or not camera_running.get(camera_id, False):
        raise HTTPException(
            status_code=404, 
            detail=f"Camera {camera_id} stream is not running. Please start the camera stream first."
        )
    
    # Check if frames are available
    if camera_id not in latest_frames:
        raise HTTPException(
            status_code=404, 
            detail=f"No frame available for camera {camera_id}. Stream may still be initializing. Please wait a moment and try again."
        )
    
    try:
        with frame_locks.get(camera_id, Lock()):
            frame_data = latest_frames.get(camera_id)
            if not frame_data or len(frame_data) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame data is empty for camera {camera_id}. Stream may still be initializing."
                )
        return Response(content=frame_data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error serving frame for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving frame: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    """Release the webcam when the application shuts down."""
    if cap is not None:
        cap.release()

def save_event_to_db(class_name, model_type, camera_id):
    """Save detection event to database with error handling"""
    db = SessionLocal()
    try:
        # Get camera info
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_name = camera.source_name if camera else f"Camera {camera_id}"
        
        event = DetectionEvent(
            class_name=class_name,
            model_type=model_type,
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=datetime.utcnow()
        )
        db.add(event)
        db.commit()
        logging.info(f"Successfully saved event: {class_name} from {camera_name}")
    except Exception as e:
        logging.error(f"Error saving event to database: {str(e)}")
        db.rollback()
    finally:
        db.close()

@app.post("/api/create_camera")
async def create_camera(
    camera_data: Dict[str, Any] = Body(..., example={
        "source_name": "Camera 1",
        "stream_type": "RTSP",
        "stream": "rtsp://example.com/stream",
        "location": "Main Entrance"
    })
):
    """Create a new camera entry in the database"""
    db = SessionLocal()
    try:
        app_logger.info(f"Received camera data: {camera_data}")
        
        # Validate required fields
        if not camera_data.get('source_name'):
            raise HTTPException(status_code=400, detail="source_name is required")
        if not camera_data.get('stream_type'):
            raise HTTPException(status_code=400, detail="stream_type is required")
        if not camera_data.get('stream'):
            raise HTTPException(status_code=400, detail="stream is required")
        
        # Create new Camera instance
        db_camera = Camera(
            source_name=camera_data['source_name'],
            stream_type=camera_data['stream_type'],
            stream=camera_data['stream'],
            location=camera_data.get('location')
        )
        
        db.add(db_camera)
        db.commit()
        db.refresh(db_camera)
        
        result = {
            "id": db_camera.id,
            "source_name": db_camera.source_name,
            "stream_type": db_camera.stream_type,
            "stream": db_camera.stream,
            "location": db_camera.location,
            "created_at": db_camera.created_at.strftime("%Y-%m-%d %H:%M:%S") if db_camera.created_at else None
        }
        app_logger.info(f"Camera created successfully: {result}")
        return result
        
    except IntegrityError as e:
        db.rollback()
        app_logger.error(f"Database integrity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with name '{camera_data.get('source_name')}' already exists"
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        app_logger.error(f"Error creating camera: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating camera: {str(e)}"
        )
    finally:
        db.close()

@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: int, camera_data: Dict[str, Any] = Body(...)):
    """Update a camera entry in the database"""
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Update fields
        if 'source_name' in camera_data:
            camera.source_name = camera_data['source_name']
        if 'stream_type' in camera_data:
            camera.stream_type = camera_data['stream_type']
        if 'stream' in camera_data:
            camera.stream = camera_data['stream']
        if 'location' in camera_data:
            camera.location = camera_data.get('location')
        
        db.commit()
        db.refresh(camera)
        
        result = {
            "id": camera.id,
            "source_name": camera.source_name,
            "stream_type": camera.stream_type,
            "stream": camera.stream,
            "location": camera.location,
            "created_at": camera.created_at.strftime("%Y-%m-%d %H:%M:%S") if camera.created_at else None
        }
        app_logger.info(f"Camera updated successfully: {result}")
        return result
        
    except IntegrityError as e:
        db.rollback()
        app_logger.error(f"Database integrity error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with name '{camera_data.get('source_name')}' already exists"
        )
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        app_logger.error(f"Error updating camera: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating camera: {str(e)}"
        )
    finally:
        db.close()

@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: int):
    """Delete a camera from the database"""
    db = SessionLocal()
    try:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        db.delete(camera)
        db.commit()
        app_logger.info(f"Camera {camera_id} deleted successfully")
        return {"message": f"Camera {camera_id} deleted successfully"}
        
    except Exception as e:
        db.rollback()
        app_logger.error(f"Error deleting camera: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting camera: {str(e)}"
        )
    finally:
        db.close()

# Add more endpoints as needed for starting/stopping streams, etc.

@app.on_event("startup")
async def startup():
    init_db()  # Initialize database
    zone_cache.refresh(SessionLocal)

@app.post("/init-db")
async def initialize_database():
    """Initialize the database tables"""
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        return {"message": "Database initialized successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initializing database: {str(e)}"
        )

@app.get("/api/detection-stats/summary")
async def get_detection_stats_summary():
    db = SessionLocal()
    try:
        stats = {
            "totalDetections": db.query(func.count(DetectionEvent.id)).scalar(),
            "objectDetections": db.query(func.count(DetectionEvent.id))
                                .filter(DetectionEvent.model_type == "objectDetection").scalar(),
            "segmentations": db.query(func.count(DetectionEvent.id))
                                .filter(DetectionEvent.model_type == "segmentation").scalar(),  
            "poseEstimations": db.query(func.count(DetectionEvent.id))
                                .filter(DetectionEvent.model_type == "pose").scalar()
        }
        return stats
    finally:
        db.close()

@app.get("/api/detection-stats/classes")
async def get_detection_classes(model: str):
    db = SessionLocal()
    try:
        query = db.query(DetectionEvent.class_name.distinct())
        if model != 'all':
            query = query.filter(DetectionEvent.model_type == model)
        
        classes = query.all()
        return [class_name[0] for class_name in classes if class_name[0]]
    finally:
        db.close()

@app.get("/api/detection-stats/daily")
async def get_daily_detection_stats(model: str = 'all', class_name: str = 'all'):
    db = SessionLocal()
    try:
        today = datetime.utcnow().date()
        tomorrow = today + timedelta(days=1)
        
        query = db.query(
            func.date_trunc('hour', DetectionEvent.timestamp).label('hour'),
            func.count(DetectionEvent.id).label('count')
        )
        
        if model != 'all':
            query = query.filter(DetectionEvent.model_type == model)
        if class_name != 'all':
            query = query.filter(DetectionEvent.class_name == class_name)
            
        hourly_stats = (query
            .filter(DetectionEvent.timestamp >= today)
            .filter(DetectionEvent.timestamp < tomorrow)
            .group_by(func.date_trunc('hour', DetectionEvent.timestamp))
            .order_by(func.date_trunc('hour', DetectionEvent.timestamp))
            .all()
        )
        
        return [{"timestamp": stat.hour.isoformat(), "count": stat.count} for stat in hourly_stats]
    finally:
        db.close()

@app.get("/api/detection-stats/weekly")
async def get_weekly_detection_stats(model: str = 'all', class_name: str = 'all'):
    db = SessionLocal()
    try:
        week_ago = datetime.utcnow().date() - timedelta(days=7)
        
        query = db.query(
            func.date_trunc('day', DetectionEvent.timestamp).label('date'),
            func.count(DetectionEvent.id).label('count')
        )
        
        if model != 'all':
            query = query.filter(DetectionEvent.model_type == model)
        if class_name != 'all':
            query = query.filter(DetectionEvent.class_name == class_name)
            
        daily_stats = (query
            .filter(DetectionEvent.timestamp >= week_ago)
            .group_by(func.date_trunc('day', DetectionEvent.timestamp))
            .order_by(func.date_trunc('day', DetectionEvent.timestamp))
            .all()
        )
        
        return [{"date": stat.date.isoformat(), "count": stat.count} for stat in daily_stats]
    finally:
        db.close()

@app.get("/api/camera-threads")
async def get_camera_threads():
    """Get information about currently running camera threads"""
    if not hasattr(app, 'camera_threads_info'):
        return JSONResponse(content={"camera_threads": []})
    
    return JSONResponse(content={
        "camera_threads": list(app.camera_threads_info.values())
    })


@app.get("/api/capture-status")
def get_capture_status():
    """Return current capture objects and their status for debugging."""
    status = {}
    for cam_id, cap in video_captures.items():
        try:
            is_ffmpeg = hasattr(cap, 'proc')
            is_imutils = getattr(cap, 'is_imutils', False)
            is_open = cap.isOpened() if hasattr(cap, 'isOpened') else False
            backend = "ffmpeg" if is_ffmpeg else ("imutils" if is_imutils else "opencv")
            status[cam_id] = {
                "backend": backend,
                "is_open": bool(is_open)
            }
        except Exception as e:
            status[cam_id] = {"error": str(e)}

    return status

@app.get("/api/detection-stats/real-time")
async def get_real_time_stats():
    """Get real-time detection statistics for the last hour"""
    db = SessionLocal()
    try:
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        # Get detection rate per minute for the last hour
        minute_stats = db.query(
            func.date_trunc('minute', DetectionEvent.timestamp).label('minute'),
            func.count(DetectionEvent.id).label('count')
        ).filter(
            DetectionEvent.timestamp >= one_hour_ago
        ).group_by(
            func.date_trunc('minute', DetectionEvent.timestamp)
        ).order_by(
            func.date_trunc('minute', DetectionEvent.timestamp).desc()
        ).limit(60).all()
        
        # Get current active cameras
        active_cameras = len(getattr(app, 'camera_threads_info', {}))
        
        # Get latest detections
        latest_detections = db.query(DetectionEvent).order_by(
            DetectionEvent.timestamp.desc()
        ).limit(10).all()
        
        return {
            "detectionRate": [{"time": stat.minute.isoformat(), "count": stat.count} for stat in minute_stats],
            "activeCameras": active_cameras,
            "latestDetections": [{
                "id": d.id,
                "model_type": d.model_type,
                "class_name": d.class_name,
                "camera_name": d.camera_name,
                "timestamp": d.timestamp.isoformat()
            } for d in latest_detections]
        }
    finally:
        db.close()

@app.get("/api/detection-stats/top-classes")
async def get_top_classes(limit: int = 10, days: int = 7):
    """Get top detected classes over a specified period"""
    db = SessionLocal()
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        top_classes = db.query(
            DetectionEvent.class_name,
            func.count(DetectionEvent.id).label('count')
        ).filter(
            DetectionEvent.timestamp >= since_date,
            DetectionEvent.class_name.isnot(None)
        ).group_by(
            DetectionEvent.class_name
        ).order_by(
            func.count(DetectionEvent.id).desc()
        ).limit(limit).all()
        
        return [{
            "class_name": class_name,
            "count": count,
            "percentage": 0  # Will be calculated in frontend
        } for class_name, count in top_classes]
    finally:
        db.close()

@app.get("/api/detection-stats/camera-performance")
async def get_camera_performance():
    """Get detection statistics per camera"""
    db = SessionLocal()
    try:
        # Get all cameras
        cameras = db.query(Camera).all()
        
        camera_stats = []
        for camera in cameras:
            # Get detection count for this camera
            detection_count = db.query(func.count(DetectionEvent.id)).filter(
                DetectionEvent.camera_id == camera.id
            ).scalar()
            
            # Get last detection time
            last_detection = db.query(DetectionEvent.timestamp).filter(
                DetectionEvent.camera_id == camera.id
            ).order_by(DetectionEvent.timestamp.desc()).first()
            
            # Check if camera is active
            is_active = camera.id in getattr(app, 'camera_threads_info', {})
            
            camera_stats.append({
                "id": camera.id,
                "name": camera.source_name,
                "location": camera.location,
                "detectionCount": detection_count,
                "lastDetection": last_detection[0].isoformat() if last_detection else None,
                "isActive": is_active,
                "uptime": "N/A"  # Could be calculated from thread info
            })
        
        return camera_stats
    finally:
        db.close()

@app.get("/api/detection-stats/hourly-pattern")
async def get_hourly_pattern(days: int = 30):
    """Get average detection patterns by hour of day"""
    db = SessionLocal()
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Get average detections by hour
        hourly_avg = db.query(
            func.extract('hour', DetectionEvent.timestamp).label('hour'),
            func.count(DetectionEvent.id).label('total_count')
        ).filter(
            DetectionEvent.timestamp >= since_date
        ).group_by(
            func.extract('hour', DetectionEvent.timestamp)
        ).order_by(
            func.extract('hour', DetectionEvent.timestamp)
        ).all()
        
        # Calculate average per hour
        hourly_pattern = []
        for hour, total_count in hourly_avg:
            avg_count = total_count / days  # Average per day for this hour
            hourly_pattern.append({
                "hour": int(hour),
                "avgCount": round(avg_count, 2)
            })
        
        return hourly_pattern
    finally:
        db.close()

@app.get("/api/detections")
async def get_detections(limit: int = 100, offset: int = 0, camera_id: Optional[int] = None):
    """Get detection history with pagination"""
    db = SessionLocal()
    try:
        query = db.query(DetectionEvent).order_by(DetectionEvent.timestamp.desc())
        
        if camera_id:
            query = query.filter(DetectionEvent.camera_id == camera_id)
        
        detections = query.offset(offset).limit(limit).all()
        
        return [{
            "id": d.id,
            "timestamp": d.timestamp.isoformat(),
            "class_name": d.class_name,
            "camera_id": d.camera_id,
            "camera_name": d.camera_name,
            "model_type": d.model_type
        } for d in detections]
    finally:
        db.close()

# Add these new endpoints for individual camera control

@app.post("/start_camera_stream/{camera_id}")
async def start_camera_stream(camera_id: int, request: ModelRequest):
    """Start stream for a specific camera."""
    global model
    
    try:
        # Set up model based on request
        if request.model_type == 'objectDetection':
            model = YOLO('yolov8m.pt')
        elif request.model_type == 'segmentation':
            model = YOLO('yolov8m-seg.pt')
        elif request.model_type == 'pose':
            model = YOLO('yolov8m-pose.pt')
        else:
            raise HTTPException(status_code=400, detail="Invalid model selected")

        # Get the specific camera
        db = SessionLocal()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        db.close()

        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        if camera_id not in camera_threads or not camera_threads[camera_id].is_alive():
            # Initialize camera resources: choose capture source depending on camera type
            try:
                # Prefer imutils VideoStream wrapper for better throughput
                try:
                    if (camera.stream_type or '').lower() == 'live':
                        app_logger.info(f"Attempting to open local webcam via imutils for camera {camera_id}")
                        cap_obj = ImutilsVideoStreamWrapper(0)
                    else:
                        app_logger.info(f"Attempting to open RTSP via imutils for camera {camera_id}. URL: {getattr(camera, 'stream', None)}")
                        cap_obj = ImutilsVideoStreamWrapper(camera.stream)
                except Exception as e:
                    app_logger.warning(f"Imutils open failed for camera {camera_id}: {e}")
                    cap_obj = None

                if not cap_obj or not cap_obj.isOpened():
                    # Try FFmpeg fallback for RTSP
                    if (camera.stream_type or '').lower() != 'live':
                        ff = RTSPFFmpegCapture(camera.stream)
                        if ff.isOpened():
                            video_captures[camera_id] = ff
                            app_logger.info(f"Camera {camera_id} opened via FFmpeg fallback (proc started)")
                        else:
                            app_logger.error(f"FFmpeg fallback not opened for camera {camera_id}. proc stdout: {bool(getattr(ff, 'stdout', None))}")
                            app_logger.error(f"Failed to open capture for camera {camera_id}. URL: {getattr(camera, 'stream', None)}")
                            raise HTTPException(status_code=500, detail=f"Failed to open capture for camera {camera_id}")
                    else:
                        app_logger.error(f"Failed to open capture for camera {camera_id}. URL: {getattr(camera, 'stream', None)}")
                        raise HTTPException(status_code=500, detail=f"Failed to open capture for camera {camera_id}")
                else:
                    video_captures[camera_id] = cap_obj
                    app_logger.info(f"Camera {camera_id} opened via imutils (isOpened={cap_obj.isOpened()})")
            except Exception as e:
                app_logger.error(f"Failed to open capture for camera {camera_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to open capture: {e}")

            frame_locks[camera_id] = Lock()
            camera_running[camera_id] = True
            detected_objects_this_session[camera_id] = set()
            
            # Create and start thread for this camera
            thread = threading.Thread(
                target=capture_frames_for_camera,
                args=(camera_id,),
                daemon=True
            )
            thread.start()
            camera_threads[camera_id] = thread
            
            app_logger.info(f"► Started Camera {camera_id}")
        
        # Increment active cameras metric
        metrics.active_cameras.inc()
        return {"message": f"Camera {camera_id} stream started", "model_type": request.model_type}
    
    except Exception as e:
        metrics.record_error(
            camera_id=str(camera_id),
            error_type=type(e).__name__,
            component="camera_start"
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_camera_stream/{camera_id}")
async def stop_camera_stream(camera_id: int):
    """Stop stream for a specific camera."""
    try:
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            camera_name = camera.source_name if camera else f"Camera {camera_id}"
        finally:
            db.close()

        if camera_id in camera_running:
            app_logger.info(f"Stopping {camera_name} (Thread ID: {camera_threads[camera_id].ident})")
            camera_running[camera_id] = False
            
            if camera_id in camera_threads:
                thread = camera_threads[camera_id]
                thread.join(timeout=2.0)
                
                if not thread.is_alive():
                    app_logger.info("╔════════════════════════════════════")
                    app_logger.info(f"║ {camera_name} Thread Stopped")
                    app_logger.info(f"║ Thread ID {thread.ident} confirmed dead")
                    app_logger.info("╚════════════════════════════════════")
                    
                    # Clean up resources
                    if camera_id in video_captures:
                        video_captures[camera_id].release()
                        del video_captures[camera_id]
                        del frame_locks[camera_id]
                        del latest_frames[camera_id]
                        del camera_threads[camera_id]
                        del camera_running[camera_id]
                else:
                    app_logger.warning(f"Thread for {camera_name} did not stop properly!")

        # Decrement active cameras metric
        metrics.active_cameras.dec()
        return {"message": f"{camera_name} stream stopped"}
    
    except Exception as e:
        metrics.record_error(
            camera_id=str(camera_id),
            error_type=type(e).__name__,
            component="camera_stop"
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_camera/{camera_id}")
async def start_camera(camera_id: int, camera_name: str = "default"):
    try:
        # Your existing camera start code...
        
        # Update metrics
        metrics.camera_status.labels(
            camera_id=str(camera_id),
            camera_name=camera_name
        ).set(1)
        metrics.active_cameras.inc()
        
        return {"status": "success", "message": f"Camera {camera_id} started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/stop_camera/{camera_id}")
async def stop_camera(camera_id: int, camera_name: str = "default"):
    try:
        # Your existing camera stop code...
        
        # Update metrics
        metrics.camera_status.labels(
            camera_id=str(camera_id),
            camera_name=camera_name
        ).set(0)
        metrics.active_cameras.dec()
        
        return {"status": "success", "message": f"Camera {camera_id} stopped"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
