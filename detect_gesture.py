import cv2
import joblib
from collections import deque
import time
from collections import Counter

from hand_utils import init_landmarker, extract_features, draw_hand_landmarks
from motion_utils import MotionDetector
from camera_utils import open_camera
from hand_utils import draw_face_landmarks
from hand_utils import detect_emotion

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Load model + scaler
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

cap, cam_index = open_camera(0)
if cap is None:
    raise RuntimeError(
        "Could not open a webcam (tried indices 0-3). "
        "Check permissions to /dev/video*, or try a different camera index."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Init landmarker (after camera is known-good)
landmarkers = init_landmarker()

prediction_buffer = deque(maxlen=15)
sentence = ""
last_added = ""
motion = MotionDetector()

print(f"Starting real-time gesture recognition (camera index {cam_index})...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features + get MediaPipe result (single pass)
        timestamp = int(time.time() * 1000)
        features, hand_result, face_result = extract_features(frame, landmarkers, timestamp, True)

        # Draw landmarks for visibility/debug
        draw_hand_landmarks(frame, hand_result)
        draw_face_landmarks(frame, face_result)

        # Motion gestures (swipe) for sentence actions
        motion_action = motion.update(hand_result, timestamp)
        if motion_action == "SPACE":
            sentence += " "
            prediction_buffer.clear()
            last_added = ""
        elif motion_action == "DELETE":
            sentence = sentence[:-1]
            prediction_buffer.clear()
            last_added = ""

        if face_result and face_result.face_landmarks:
            cv2.putText(frame, "FACE DETECTED",
                (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 2)
        else:
            cv2.putText(frame, "NO FACE",
                (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)
        if features is None:
            cv2.putText(frame, "NO HAND DETECTED",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        else:
            # Scale features (IMPORTANT)
            features = scaler.transform([features])[0]

            # Predict
            gesture = model.predict([features])[0]
            prediction_buffer.append(gesture)

            # Smooth prediction
            if len(prediction_buffer) == prediction_buffer.maxlen:
                stable_gesture = Counter(prediction_buffer).most_common(1)[0][0]

                if stable_gesture != last_added:
                    if stable_gesture == "SPACE":
                        sentence += " "
                    elif stable_gesture == "DELETE":
                        sentence = sentence[:-1]
                    else:
                        sentence += stable_gesture

                    last_added = stable_gesture

            cv2.putText(frame, f"Gesture: {gesture}",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.putText(frame, f"Sentence: {sentence}",
                    (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

        cv2.imshow("Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            sentence = ""
            last_added = ""
        if key == ord('q'):
            break
finally:
    cap.release()
    try:
        landmarker.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
