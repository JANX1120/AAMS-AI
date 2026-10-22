"""HTTP image upload API for the AAMS detection pipeline."""

import asyncio
import logging
import os
from time import perf_counter
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

if __package__:
    from .detector import detect_frame, initialize_models
else:
    from detector import detect_frame, initialize_models

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
CAMERA_SESSION_TIMEOUT = 60.0


@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(initialize_models)

    # Prevent overlapping AI inference requests.
    app.state.inference_lock = asyncio.Lock()

    # Track which camera currently owns the detector.
    app.state.active_camera_session = None

    # Time of the last accepted camera frame.
    app.state.last_camera_activity = 0.0

    yield


app = FastAPI(title="AAMS AI Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("AAMS_CORS_ORIGINS", "*").split(",") if origin.strip()],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"],
)


def error_response(status, message):
    return JSONResponse(status_code=status, content={"success": False, "message": message})


@app.get("/")
def home():
    return {
        "success": True, "message": "AAMS AI Detection Server is running.",
        "models": {"object_detection": "YOLO", "head_pose": "6DRepNet"},
    }


@app.get("/health")
def health():
    return {"success": True, "status": "online"}


def decode_and_detect(image_bytes, session_id):
    try:
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return error_response(400, "Invalid image.")
    if frame is None:
        return error_response(400, "Invalid image.")
    if frame.shape[0] * frame.shape[1] > MAX_IMAGE_PIXELS:
        return error_response(413, "Image exceeds the 16 megapixel limit.")
    started = perf_counter()
    result = detect_frame(frame, session_id=session_id)
    result["inference_ms"] = round((perf_counter() - started) * 1000, 1)
    return result


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    session_id: str = Form(..., min_length=1, max_length=128),
):
    try:
        # Browser blobs may arrive as application/octet-stream.
        if file.content_type and not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):
            return error_response(415, "Upload an image file.")
        image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if not image_bytes:
            return error_response(400, "No image data was received.")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return error_response(413, "Image exceeds the 10 MB upload limit.")
        # Reject overlapping inference rather than building an unbounded queue
        # of stale camera frames. Health checks remain responsive.
        if app.state.inference_lock.locked():
            return error_response(503, "Detector is busy. Retry with the next frame.")

        async with app.state.inference_lock:
            now = perf_counter()

            active_session = app.state.active_camera_session
            last_activity = app.state.last_camera_activity

            # Reject a different camera while the current session is active.
            if (
                active_session is not None
                and active_session != session_id
                and now - last_activity < CAMERA_SESSION_TIMEOUT
            ):
                return error_response(
                    409,
                    "Another camera session is active. Please try again later."
                )

            result = await run_in_threadpool(
                decode_and_detect,
                image_bytes,
                session_id
            )

            # Update ownership only after successful detection.
            if isinstance(result, dict) and result.get("success") is True:
                app.state.active_camera_session = session_id
                app.state.last_camera_activity = perf_counter()

            return result
    except Exception:
        logger.exception("Detection request failed")
        return error_response(500, "Detection failed. Please try again.")
    finally:
        await file.close()
