import cv2

from src.pose_detector import PoseDetector
from src.measurements import calculate_measurements


image = cv2.imread("test_images/person.jpg")

if image is None:
    raise FileNotFoundError("Could not load test image.")

# Detect pose
detector = PoseDetector()
landmarks = detector.detect(image)

if landmarks is None:
    print("No person detected.")
else:
    print("Person detected!")

    # Calculate body proportions
    measurements = calculate_measurements(landmarks)

    print("\nBody Proportions:")
    for name, value in measurements.items():
        print(f"{name}: {value:.4f}")