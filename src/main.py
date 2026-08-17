import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.measurements import calculate_measurements
from src.pose_detector import PoseDetector


app = FastAPI(
    title="AI Body Analysis API",
    description="Pose detection and relative body proportion analysis API",
    version="1.0.0",
)

pose_detector = PoseDetector()


# ---------------------------------------------------------
# Global 500 Internal Server Error Handler
# ---------------------------------------------------------
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        },
    )


# ---------------------------------------------------------
# Health Endpoints
# ---------------------------------------------------------
@app.get(
    "/api/ai/health",
    responses={
        500: {"description": "Internal server error"},
    },
)
def health_check():
    return {
        "status": "healthy",
        "model": "pose-v1",
    }


@app.get(
    "/health",
    responses={
        500: {"description": "Internal server error"},
    },
)
def health():
    return {
        "status": "healthy",
        "model": "pose-v1",
    }


# ---------------------------------------------------------
# Pose Detection Endpoint
# ---------------------------------------------------------
@app.post(
    "/api/ai/pose",
    responses={
        400: {
            "description": "Invalid image file or unreadable image"
        },
        404: {
            "description": "No person detected in the image"
        },
        422: {
            "description": "Required image file was not provided"
        },
        500: {
            "description": "Internal server error"
        },
    },
)
async def detect_pose(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    # Read uploaded file
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Convert uploaded bytes to OpenCV image
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image.",
        )

    # Detect pose landmarks
    landmarks = pose_detector.detect(image)

    processing_time = round(
        time.perf_counter() - start_time,
        4,
    )

    # No person detected
    if landmarks is None:
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    return {
        "person_detected": True,
        "landmarks": landmarks,
        "measurements": {},
        "processing_time": processing_time,
    }


# ---------------------------------------------------------
# Body Measurements Endpoint
# ---------------------------------------------------------
@app.post(
    "/api/ai/measurements",
    responses={
        400: {
            "description": "Invalid image file or unreadable image"
        },
        404: {
            "description": "No person detected in the image"
        },
        422: {
            "description": "Required image file was not provided"
        },
        500: {
            "description": "Internal server error"
        },
    },
)
async def detect_measurements(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file.",
        )

    # Read uploaded file
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Convert uploaded bytes to OpenCV image
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image.",
        )

    # Detect landmarks
    landmarks = pose_detector.detect(image)

    processing_time = round(
        time.perf_counter() - start_time,
        4,
    )

    # No person detected
    if landmarks is None:
        raise HTTPException(
            status_code=404,
            detail="No person detected in the image.",
        )

    # Calculate relative body proportions
    measurements = calculate_measurements(landmarks)

    return {
        "person_detected": True,
        "landmarks": landmarks,
        "measurements": measurements,
        "processing_time": processing_time,
    }