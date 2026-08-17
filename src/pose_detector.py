import cv2
import mediapipe as mp


class PoseDetector:
    def __init__(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path="models/pose_landmarker_full.task"
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        )

    def detect(self, image):
        if image is None:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        LANDMARK_NAMES = {
                 0: "nose",
                 11: "left_shoulder",
                 12: "right_shoulder",
                 13: "left_elbow",
                 14: "right_elbow",
                 15: "left_wrist",
                 16: "right_wrist",
                 23: "left_hip",
                 24: "right_hip",
                 25: "left_knee",
                 26: "right_knee",
                 27: "left_ankle",
                 28: "right_ankle",
                 }

        landmarks = {}

        for index, landmark in enumerate(result.pose_landmarks[0]):
            if index in LANDMARK_NAMES:
                landmarks[LANDMARK_NAMES[index]] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }

        return landmarks