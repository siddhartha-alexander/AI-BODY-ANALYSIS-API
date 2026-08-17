# AI Body Analysis API

An AI/ML API that accepts a human body image, detects pose landmarks using MediaPipe, calculates relative body proportions, and returns the results as JSON through a FastAPI server.

## Project Pipeline

Image
→ OpenCV
→ MediaPipe Pose Landmarker
→ Body Landmarks
→ Relative Measurements
→ FastAPI
→ JSON Response

## Features

- Human pose detection using MediaPipe
- Landmark coordinate extraction
- Relative shoulder, hip, arm, leg, and torso measurements
- Shoulder-to-hip ratio
- FastAPI REST endpoints
- JSON responses
- Input validation and error handling
- Swagger API documentation

## Project Structure

```text
AI-BODY-ANALYSIS-API/
│
├── models/
│   └── pose_landmarker_full.task
│
├── src/
│   ├── pose_detector.py
│   ├── measurements.py
│   └── main.py
│
├── test_images/
│   └── person.jpg
│
├── test_pose.py
├── test_measurements.py
├── requirements.txt
├── API_DOCUMENTATION.md
└── README.md