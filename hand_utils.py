import numpy as np
import cv2
from tasks_utils import init_hand_landmarker, detect_hands, extract_both_hand_keypoints
from tasks_utils import init_face_landmarker, detect_face



# =============================
# INIT
# =============================
def init_landmarker():
    hand = init_hand_landmarker("hand_landmarker.task")
    face = init_face_landmarker("face_landmarker.task")
    return hand, face


# =============================
# NORMALIZE
# =============================
def normalize_keypoints(keypoints, frame):
    keypoints = keypoints.reshape(-1, 3)
    keypoints[:, 0] /= frame.shape[1]
    keypoints[:, 1] /= frame.shape[0]
    return keypoints.flatten()


# =============================
# FEATURE PIPELINE
# =============================
def extract_features(frame, landmarkers, timestamp, return_result=False):
    hand_landmarker, face_landmarker = landmarkers

    hand_result = detect_hands(frame, hand_landmarker, timestamp)
    face_result = detect_face(frame, face_landmarker, timestamp)

    # =============================
    # HAND FEATURES
    # =============================
    hand_keypoints = extract_both_hand_keypoints(hand_result)

    if hand_keypoints is None:
        hand_features = np.zeros(126)  # 21 landmarks * 3 * 2 hands
    else:
        hand_features = normalize_keypoints(hand_keypoints, frame)

    # =============================
    # FACE FEATURES
    # =============================
    face_features = [0, 0, 0]

    if face_result and face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]

        mouth = abs(face_landmarks[13].y - face_landmarks[14].y)
        tilt = face_landmarks[33].y - face_landmarks[263].y
        eye_avg = (face_landmarks[33].y + face_landmarks[263].y) / 2
        head_ud = face_landmarks[1].y - eye_avg

        face_features = [mouth, tilt, head_ud]

    # =============================
    # COMBINE FEATURES
    # =============================
    features = np.concatenate([hand_features, face_features])

    if return_result:
        return features, hand_result, face_result

    return features

    def detect_emotion(face_result):
        if face_result is None or not face_result.face_landmarks:
            return "No Face"

        face = face_result.face_landmarks[0]

    # Key points
        mouth_open = abs(face[13].y - face[14].y)
        left_eye = face[33].y
        right_eye = face[263].y
        tilt = left_eye - right_eye

    # Simple rules
        if mouth_open > 0.03:
            return "Surprised"

        if tilt > 0.02:
            return "Tilt Left"

        if tilt < -0.02:
            return "Tilt Right"

        return "Neutral"
# =============================
# DRAW LANDMARKS (VISIBLE)
# =============================
def draw_hand_landmarks(frame, result):
    if result is None or not result.hand_landmarks:
        return

    h, w, _ = frame.shape

    for hand in result.hand_landmarks:
        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            
            
# face drawing function #
def draw_face_landmarks(frame, face_result):
    if face_result is None or not face_result.face_landmarks:
        return

    h, w, _ = frame.shape

    for face in face_result.face_landmarks:
        for lm in face:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)
    
    # =============================
# SIMPLE EMOTION DETECTION
# =============================
def detect_emotion(face_result):
    if face_result is None or not face_result.face_landmarks:
        return "No Face"

    face = face_result.face_landmarks[0]

    # Mouth openness
    mouth = abs(face[13].y - face[14].y)

    # Eyebrow vs eye distance (rough emotion cue)
    eyebrow = face[70].y
    eye = face[159].y
    brow_eye_dist = abs(eyebrow - eye)

    # Simple rules (you can improve later)
    if mouth > 0.05:
        return "Surprised 😲"
    elif brow_eye_dist < 0.015:
        return "Angry 😠"
    else:
        return "Neutral 😐"
            
