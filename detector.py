"""Object tracking and per-frame head-pose signals for the AAMS dashboard."""

import logging
import os
from functools import wraps
from pathlib import Path
from threading import RLock

import numpy as np
import torch
from sixdrepnet import SixDRepNet
from ultralytics import YOLO

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
YOLO_MODEL_PATH = BASE_DIR / "yolo26n.pt"
YOLO_CONFIDENCE = 0.50
MIN_OBJECT_CONFIDENCE = 0.60
HEAD_YAW_THRESHOLD = 35
HEAD_PITCH_THRESHOLD = 30
MIN_PERSON_WIDTH = 100
MIN_PERSON_HEIGHT = 160

MIN_HEAD_CROP_WIDTH = 80
MIN_HEAD_CROP_HEIGHT = 80

YOLO_IMAGE_SIZE = 640

# Ultralytics' predictor and persistent tracker are mutable. Protect the entire
# pipeline, including initialization, when called from multiple worker threads.
_model_lock = RLock()
yolo_model = None
head_pose_model = None

# Multi-frame head-turn observation tracking
HEAD_TURN_REQUIRED_FRAMES = 3
HEAD_TURN_MAX_GAP_FRAMES = 2

head_turn_history = {}
frame_counter = 0
active_session_id = None


def initialize_models():
    """Load once, using local weights and CPU head-pose inference."""
    global yolo_model, head_pose_model
    with _model_lock:
        if yolo_model is None:
            # Limit CPU oversubscription on small hosted instances. Override for
            # the deployment CPU allocation with AAMS_TORCH_THREADS.
            threads = int(os.getenv("AAMS_TORCH_THREADS", str(min(4, torch.get_num_threads()))))
            if threads < 1:
                raise ValueError("AAMS_TORCH_THREADS must be positive.")
            torch.set_num_threads(threads)
            if not YOLO_MODEL_PATH.is_file():
                raise FileNotFoundError(f"YOLO weights are missing: {YOLO_MODEL_PATH}")
            yolo_model = YOLO(str(YOLO_MODEL_PATH))
        if head_pose_model is None:
            head_pose_model = SixDRepNet(gpu_id=-1)


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _model_lock:
            return function(*args, **kwargs)
    return wrapped


def get_head_direction(pitch, yaw, roll):
    """Preserve the dashboard's existing angle and direction convention."""
    pitch, yaw, roll = float(pitch), float(yaw), float(roll)
    if not np.isfinite([pitch, yaw, roll]).all():
        raise ValueError("Head-pose angles must be finite.")
    if yaw <= -HEAD_YAW_THRESHOLD:
        return "left"
    if yaw >= HEAD_YAW_THRESHOLD:
        return "right"
    if pitch <= -HEAD_PITCH_THRESHOLD:
        return "up"
    if pitch >= HEAD_PITCH_THRESHOLD:
        return "down"
    return "forward"


def detect_head_pose(frame, person_boxes):
    """Estimate pose from approximate head crops; these are possible signals only."""
    head_results = []
    height, width = frame.shape[:2]
    for index, person in enumerate(person_boxes):
        track_id = person.get("track_id")
        label = f"Person {track_id if track_id is not None else index + 1}"
        x1, y1 = max(0, int(person["x1"])), max(0, int(person["y1"]))
        x2, y2 = min(width, int(person["x2"])), min(height, int(person["y2"]))
        person_width, person_height = x2 - x1, y2 - y1
        if person_width < MIN_PERSON_WIDTH or person_height < MIN_PERSON_HEIGHT:
            continue
        head_x1 = x1 + int(person_width * 0.15)
        head_x2 = x2 - int(person_width * 0.15)
        head_y2 = y1 + int(person_height * 0.40)
        crop = frame[y1:head_y2, head_x1:head_x2]
        if crop.size == 0:
            continue

        crop_height, crop_width = crop.shape[:2]

        if (
            crop_width < MIN_HEAD_CROP_WIDTH
            or crop_height < MIN_HEAD_CROP_HEIGHT
        ):
            continue
        try:
            angles = head_pose_model.predict(np.ascontiguousarray(crop))
            pitch, yaw, roll = [float(np.asarray(angle).reshape(-1)[0]) for angle in angles]
            direction = get_head_direction(pitch, yaw, roll)
            head_results.append({
                "person": label, "track_id": track_id,
                "pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2),
                "direction": direction,
                "possible_peeking": False,
                "head_turned": direction in {"left", "right"},
                "box": {"x1": head_x1, "y1": y1, "x2": head_x2, "y2": head_y2},
            })
        except Exception:
            logger.exception("Head-pose estimation failed for %s", label)
    return head_results

def update_head_turn_history(heads):
    """Track repeated head turns for a single camera stream."""
    global frame_counter

    frame_counter += 1
    observations = []

    for head in heads:
        track_id = head.get("track_id")

        # Do not combine observations without a reliable tracking ID.
        if track_id is None:
            continue

        direction = head.get("direction", "forward")
        turned = direction in {"left", "right"}

        previous = head_turn_history.get(track_id)

        if turned:
            if (
                previous
                and previous["direction"] == direction
                and frame_counter - previous["last_seen"] == 1
            ):
                count = previous["count"] + 1
            else:
                count = 1
        else:
            count = 0

        head_turn_history[track_id] = {
            "direction": direction,
            "count": count,
            "last_seen": frame_counter,
        }

        observations.append({
            "person": head["person"],
            "track_id": track_id,
            "direction": direction,
            "consecutive_frames": count,
            "sustained_head_turn": turned and count >= HEAD_TURN_REQUIRED_FRAMES,
        })

    # Remove stale tracking records.
    stale_ids = [
        track_id
        for track_id, state in head_turn_history.items()
        if frame_counter - state["last_seen"] > HEAD_TURN_MAX_GAP_FRAMES
    ]

    for track_id in stale_ids:
        del head_turn_history[track_id]

    return observations


@_serialized
@torch.inference_mode()
def detect_frame(frame, session_id=None):
    """Track frames from ONE ordered camera stream per server process.

    Persistent IDs must not be shared between unrelated cameras. A multi-camera
    deployment needs separate tracker instances and an explicit stream ID.
    """

    global active_session_id, frame_counter

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("A valid camera session ID is required.")

    if session_id != active_session_id:
        logger.info("Switching AI monitoring session.")

        if yolo_model is not None:
            predictor = getattr(yolo_model, "predictor", None)
            if predictor is not None:
                for tracker in getattr(predictor, "trackers", []):
                    tracker.reset()

        head_turn_history.clear()
        frame_counter = 0
        active_session_id = session_id

    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
        raise ValueError("Expected a non-empty uint8 BGR image.")
    initialize_models()
    height, width = frame.shape[:2]
    results = yolo_model.track(
        source=frame, conf=YOLO_CONFIDENCE, classes=[0, 67, 73],
        persist=True, tracker="bytetrack.yaml", verbose=False, imgsz=YOLO_IMAGE_SIZE,
    )
    counts = {"person": 0, "cell phone": 0, "book": 0}
    detections, people = [], []
    boxes = results[0].boxes if results else None
    for box in boxes if boxes is not None else ():
        class_id = int(box.cls[0])
        name = yolo_model.names[class_id]
        confidence = float(box.conf[0])
        threshold = YOLO_CONFIDENCE if name == "person" else MIN_OBJECT_CONFIDENCE
        if name not in counts or not np.isfinite(confidence) or confidence < threshold:
            continue
        coordinates = np.asarray(box.xyxy[0].tolist(), dtype=float)
        if not np.isfinite(coordinates).all():
            continue
        x1, y1, x2, y2 = coordinates
        x1, x2 = [int(round(np.clip(x, 0, width))) for x in (x1, x2)]
        y1, y2 = [int(round(np.clip(y, 0, height))) for y in (y1, y2)]
        if x2 <= x1 or y2 <= y1:
            continue
        bounds = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        detection = {"class": name, "confidence": round(confidence, 2), "box": bounds}
        counts[name] += 1
        if name == "person":
            track_id = int(box.id[0]) if box.id is not None else None
            detection.update(track_id=track_id, person=f"Person {track_id if track_id is not None else counts[name]}")
            people.append({"track_id": track_id, **bounds})
        detections.append(detection)

    heads = detect_head_pose(frame, people)

    # Analyze head movements across consecutive frames.
    head_turn_tracking = update_head_turn_history(heads)

    # Record head-direction observations for future
    # multi-frame analysis.
    head_turn_observations = [
        {
            "person": head["person"],
            "track_id": head["track_id"],
            "direction": head["direction"],
            "yaw": head["yaw"],
            "pitch": head["pitch"],
        }
        for head in heads
        if head.get("head_turned", False)
    ]

    # Do not classify head turns as cheating.
    peeking_people = []
    violations = []
    if counts["cell phone"]:
        violations.append("Mobile Phone Detected")
    if counts["book"]:
        violations.append("Unauthorized Book Detected")
    if peeking_people:
        violations.append("Possible Peeking Detected")
    return {
        "success": True, "people": counts["person"],
        "phones": counts["cell phone"], "books": counts["book"],
        "possible_peeking": bool(peeking_people), "peeking_count": len(peeking_people),
        "peeking_people": peeking_people,
        "head_pose": heads,
        "head_turn_observations": head_turn_observations,
        "head_turn_tracking": head_turn_tracking,
        "violations": len(violations), "violation_list": violations,
        "detections": detections, "image_width": width, "image_height": height,
    }
