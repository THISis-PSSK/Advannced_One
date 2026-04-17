import streamlit as st
import cv2
import time
import joblib
import numpy as np
from collections import Counter, deque

from hand_utils import init_landmarker, extract_features, draw_hand_landmarks, draw_face_landmarks, detect_emotion
from camera_utils import open_camera

# =============================
# LOAD MODEL
# =============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

landmarker = init_landmarker()

# =============================
# UI
# =============================
st.set_page_config(layout="wide")
st.title("🖐️ ISL Gesture + Emotion Recognition")

col1, col2 = st.columns([2, 1])

frame_placeholder = col1.empty()
text_placeholder = col2.empty()
diag_placeholder = col2.empty()

# =============================
# SESSION STATE
# =============================
if "running" not in st.session_state:
    st.session_state.running = False

if "buffer" not in st.session_state:
    st.session_state.buffer = deque(maxlen=10)
    st.session_state.sentence = ""
    st.session_state.last_word = ""
    st.session_state.feature_len = None

# =============================
# BUTTONS
# =============================
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("Start Camera"):
        st.session_state.running = True

with col_btn2:
    if st.button("Stop Camera"):
        st.session_state.running = False

# =============================
# CAMERA LOOP
# =============================
if st.session_state.running:

    cap, _ = open_camera(0)
    
    while st.session_state.running:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not working")

        else:
            #frame = cv2.flip(frame, 1)
            timestamp = int(time.time() * 1000)
    
            features, hand_result, face_result = extract_features(
                frame, landmarker, timestamp, True
            )
    
            draw_hand_landmarks(frame, hand_result)
            draw_face_landmarks(frame, face_result)

            emotion = detect_emotion(face_result)

        # =============================
        # GESTURE
        # =============================
        if features is not None:

            if st.session_state.feature_len is None:
                st.session_state.feature_len = len(features)

            if len(features) == st.session_state.feature_len:
                scaled = scaler.transform([features])[0]
                gesture = model.predict([scaled])[0]

                st.session_state.buffer.append(gesture)

                if len(st.session_state.buffer) == st.session_state.buffer.maxlen:
                    stable = Counter(st.session_state.buffer).most_common(1)[0][0]

                    if stable != st.session_state.last_word:
                        st.session_state.sentence += stable + " "
                        st.session_state.last_word = stable

        # =============================
        # DISPLAY
        # =============================
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        text_placeholder.markdown(f"### 📝 Sentence:\n{st.session_state.sentence}")

        diag_placeholder.markdown(f"""
        **Emotion:** {emotion}  
        **Buffer:** {list(st.session_state.buffer)}
        """)

    cap.release()

    # 🔁 Auto-refresh for live camera
    #st.rerun()
if st.session_state.running:

    cap, _ = open_camera(0)

    while st.session_state.running:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.flip(frame, 1)
        timestamp = int(time.time() * 1000)

        features, hand_result, face_result = extract_features(
            frame, landmarker, timestamp, True
        )

        draw_hand_landmarks(frame, hand_result)
        draw_face_landmarks(frame, face_result)

        emotion = detect_emotion(face_result)

        # gesture logic same...

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame)

        text_placeholder.markdown(f"### 📝 Sentence:\n{st.session_state.sentence}")

        diag_placeholder.markdown(f"""
        **Emotion:** {emotion}  
        **Buffer:** {list(st.session_state.buffer)}
        """)

    cap.release()
