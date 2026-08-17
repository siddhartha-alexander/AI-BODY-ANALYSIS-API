import math


def distance(point1, point2):
    """
    Calculate 2D distance between two landmarks.
    """
    return math.sqrt(
        (point2["x"] - point1["x"]) ** 2
        + (point2["y"] - point1["y"]) ** 2
    )


def calculate_measurements(landmarks):
    """
    Calculate relative body proportions from pose landmarks.
    """

    required = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    # Check whether required landmarks exist
    for name in required:
        if name not in landmarks:
            raise ValueError(f"Missing landmark: {name}")

    # Shoulder width
    shoulder_width = distance(
        landmarks["left_shoulder"],
        landmarks["right_shoulder"]
    )

    # Hip width
    hip_width = distance(
        landmarks["left_hip"],
        landmarks["right_hip"]
    )

    # Left arm = shoulder → elbow + elbow → wrist
    left_arm = (
        distance(landmarks["left_shoulder"], landmarks["left_elbow"])
        + distance(landmarks["left_elbow"], landmarks["left_wrist"])
    )

    # Right arm
    right_arm = (
        distance(landmarks["right_shoulder"], landmarks["right_elbow"])
        + distance(landmarks["right_elbow"], landmarks["right_wrist"])
    )

    # Left leg = hip → knee + knee → ankle
    left_leg = (
        distance(landmarks["left_hip"], landmarks["left_knee"])
        + distance(landmarks["left_knee"], landmarks["left_ankle"])
    )

    # Right leg
    right_leg = (
        distance(landmarks["right_hip"], landmarks["right_knee"])
        + distance(landmarks["right_knee"], landmarks["right_ankle"])
    )

    # Torso = midpoint of shoulders to midpoint of hips
    shoulder_midpoint = {
        "x": (
            landmarks["left_shoulder"]["x"]
            + landmarks["right_shoulder"]["x"]
        ) / 2,
        "y": (
            landmarks["left_shoulder"]["y"]
            + landmarks["right_shoulder"]["y"]
        ) / 2,
    }

    hip_midpoint = {
        "x": (
            landmarks["left_hip"]["x"]
            + landmarks["right_hip"]["x"]
        ) / 2,
        "y": (
            landmarks["left_hip"]["y"]
            + landmarks["right_hip"]["y"]
        ) / 2,
    }

    torso = distance(shoulder_midpoint, hip_midpoint)

    # Shoulder-to-hip ratio
    shoulder_to_hip_ratio = (
        shoulder_width / hip_width
        if hip_width != 0
        else None
    )

    return {
        "shoulder_width_ratio": shoulder_width,
        "hip_width_ratio": hip_width,
        "left_arm_ratio": left_arm,
        "right_arm_ratio": right_arm,
        "left_leg_ratio": left_leg,
        "right_leg_ratio": right_leg,
        "torso_ratio": torso,
        "shoulder_to_hip_ratio": shoulder_to_hip_ratio,
    }