import base64
import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
import joblib
import numpy as np
from collections import Counter, deque

from camera_utils import open_camera
from hand_utils import (
    init_landmarker,
    extract_features,
    draw_hand_landmarks,
    draw_face_landmarks,
    detect_emotion
)
from motion_utils import MotionDetector

MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"


def to_tk_image(frame):
    frame = cv2.resize(frame, (640, 480))
    _, png = cv2.imencode(".png", frame)
    return tk.PhotoImage(data=base64.b64encode(png).decode())

class GestureUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Non-Manual Features Translator")
        self.root.geometry("1200x700")
        self.root.configure(bg="#e5e7eb")

        self.last_word = ""
        self.cap = None
        self.running = False

        # ===== HEADER =====
        header = tk.Frame(root, bg="#14b8a6", height=50)
        header.pack(fill="x")

        tk.Label(header, text="ISL Translator",
                 bg="#14b8a6", fg="white",
                 font=("Arial", 14, "bold")).pack(side="left", padx=10)

        tk.Button(header, text="Show Gestures",
                  bg="#facc15", command=self.show_gestures).pack(side="right", padx=5)

        tk.Button(header, text="Clear",
                  bg="#ef4444", command=self.clear_text).pack(side="right", padx=5)

        # ===== MAIN =====
        main = tk.Frame(root, bg="#e5e7eb")
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ===== LEFT (CAMERA) =====
        left = tk.Frame(main, bg="white", bd=2, relief="ridge")
        left.pack(side="left", fill="both", expand=True, padx=5)

        self.camera_label = tk.Label(left)
        self.camera_label.pack(padx=10, pady=10)

        btn_bar = tk.Frame(left)
        btn_bar.pack()

        tk.Button(btn_bar, text="Start Camera",
                  bg="#22c55e", command=self.start_cam).pack(side="left", padx=5)

        tk.Button(btn_bar, text="Stop",
                  bg="#ef4444", command=self.stop_cam).pack(side="left", padx=5)

        # ===== RIGHT PANEL =====
        right = tk.Frame(main, bg="white", bd=2, relief="ridge", width=300)
        right.pack(side="right", fill="y", padx=5)

        tk.Label(right, text="Detected Sentence",
                 font=("Arial", 12, "bold")).pack(pady=5)

        self.translation_box = tk.Text(right, height=6, font=("Arial", 14))
        self.translation_box.pack(fill="x", padx=10)

        tk.Label(right, text="Diagnostics",
                 font=("Arial", 11, "bold")).pack(pady=5)

        self.diagnostics = tk.Text(right, height=10)
        self.diagnostics.pack(fill="both", padx=10)

        # ===== ML =====
        self.landmarker = init_landmarker()
        self.motion = MotionDetector()

        self.model = joblib.load(MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)

        self.buffer = deque(maxlen=10)
        self.feature_len = None
        self.sentence = ""

    # ================= GESTURE LIST =================
    def get_all_gestures(self):
        import pandas as pd

        if not os.path.exists("gesture_data.csv"):
            return ["No dataset found"]

        df = pd.read_csv("gesture_data.csv", header=None)
        return sorted(set(df.iloc[:, -1].astype(str)))

    def show_gestures(self):
        gestures = self.get_all_gestures()

        win = tk.Toplevel(self.root)
        win.title("Stored Gestures")

        text = tk.Text(win, font=("Arial", 12))
        text.pack()

        for g in gestures:
            text.insert("end", f"{g}\n")

    def clear_text(self):
        self.sentence = ""
        self.last_word = ""
        self.translation_box.delete("1.0", "end")

    # ================= CAMERA =================
    def start_cam(self):
        if self.cap is None:
            self.cap, _ = open_camera(0)

        self.running = True
        self.loop()

    def stop_cam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    # ================= LOOP =================
    def loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.loop)
            return

        #frame = cv2.flip(frame, 1)
        timestamp = int(time.time() * 1000)

        features, hand_result, face_result = extract_features(
            frame, self.landmarker, timestamp, True
        )

        draw_hand_landmarks(frame, hand_result)
        draw_face_landmarks(frame, face_result)

        emotion = detect_emotion(face_result)

        # ===== DIAGNOSTICS =====
        self.diagnostics.configure(state="normal")
        self.diagnostics.delete("1.0", "end")

        self.diagnostics.insert("end", f"Emotion: {emotion}\n")
        self.diagnostics.insert("end", f"Sentence: {self.sentence}\n")

        if features is None:
            self.diagnostics.insert("end", "Hand: NOT DETECTED\n")
        else:
            self.diagnostics.insert("end", "Hand: DETECTED\n")

        self.diagnostics.configure(state="disabled")

        # ===== GESTURE =====
        if features is not None:
            if self.feature_len is None:
                self.feature_len = len(features)

            if len(features) == self.feature_len:
                scaled = self.scaler.transform([features])[0]
                gesture = self.model.predict([scaled])[0]

                self.buffer.append(gesture)

                if len(self.buffer) == self.buffer.maxlen:
                    stable = Counter(self.buffer).most_common(1)[0][0]

                    if stable != self.last_word:
                        self.sentence += stable + " "
                        self.last_word = stable

                        self.translation_box.delete("1.0", "end")
                        self.translation_box.insert("1.0", self.sentence)

        img = to_tk_image(frame)
        self.camera_label.configure(image=img)
        self.camera_label.image = img

        self.root.after(30, self.loop)

def main():
    root = tk.Tk()
    app = GestureUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
