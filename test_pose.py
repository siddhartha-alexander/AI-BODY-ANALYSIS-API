import cv2

from src.pose_detector import PoseDetector


image = cv2.imread("test_images/person.jpg")

if image is None:
    raise FileNotFoundError("Could not load test image.")

detector = PoseDetector()

landmarks = detector.detect(image)

if landmarks is None:
    print("No person detected.")
else:
    print(f"Person detected!")
    print(f"Total landmarks: {len(landmarks)}")

    for name, landmark in landmarks.items():
        print(name, landmark)