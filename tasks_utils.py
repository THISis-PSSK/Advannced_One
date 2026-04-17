import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#FACE LANDMARK INIT #
def init_face_landmarker(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    return vision.FaceLandmarker.create_from_options(options)
    
def detect_face(frame, face_landmarker, timestamp):
    image_rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    return face_landmarker.detect_for_video(mp_image, timestamp)

# =========================================================
# INIT HAND LANDMARKER (VIDEO MODE — STABLE)
# =========================================================
def init_hand_landmarker(model_path: str):
    base_options = python.BaseOptions(model_asset_path=model_path)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=vision.RunningMode.VIDEO
    )

    return vision.HandLandmarker.create_from_options(options)


# =========================================================
# DETECT HANDS (VIDEO MODE)
# =========================================================
def detect_hands(frame, landmarker, timestamp):
    image_rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    return landmarker.detect_for_video(mp_image, timestamp)


# =========================================================
# EXTRACT KEYPOINTS (126 FEATURES)
# =========================================================
def extract_both_hand_keypoints(result):
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    if result is None or not result.hand_landmarks:
        return None

    if result.hand_landmarks:
        if len(result.hand_landmarks) >= 1:
            left = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float32)

        if len(result.hand_landmarks) >= 2:
            right = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[1]], dtype=np.float32)

    return np.concatenate([left.flatten(), right.flatten()])
