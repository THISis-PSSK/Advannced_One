import cv2
import numpy as np
import pandas as pd
import joblib
import argparse
import os
import shutil
import time
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from hand_utils import (
    init_landmarker,
    extract_features,
    draw_hand_landmarks,
    draw_face_landmarks,
    detect_emotion
)
from camera_utils import open_camera

# =============================
# CONFIG
# =============================
DATA_FILE = "gesture_data.csv"
BACKUP_FILE = "gesture_data_backup.csv"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

GESTURES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "SPACE", "DELETE",
]

DEFAULT_SAMPLES_PER_LABEL = 20
DEFAULT_MOTION_LABELS = ["SPACE", "DELETE"]


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _parse_label_list(prompt, default_list=None):
    raw = input(prompt).strip()
    if not raw and default_list is not None:
        return list(default_list)
    items = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return items


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train gesture model")
    parser.add_argument(
        "--mode",
        choices=["n", "a", "t"],
        default=None,
        help="n=collect new, a=add new labels, t=train only (no collection)",
    )
    parser.add_argument("--samples-per-label", type=int, default=None)
    parser.add_argument(
        "--gestures",
        type=str,
        default=None,
        help="Comma separated gestures for mode 'a'. Example: A,B,C",
    )
    parser.add_argument(
        "--collect-motion",
        action="store_true",
        help="Also collect motion labels (added as normal labels in dataset)",
    )
    parser.add_argument(
        "--motion-labels",
        type=str,
        default=None,
        help="Comma separated motion labels. Example: SPACE,DELETE,MOVEUP",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Preferred camera index used by open_camera()",
    )

    args = parser.parse_args(argv)

    if args.mode is None:
        # -----------------------------
        # MENU (interactive)
        # -----------------------------
        print("\nPress 'n' → Collect NEW data (keeps existing dataset)")
        print("Press 'a' → Add NEW gesture(s) (append to existing dataset)")
        print("Press 't' → Train using existing data only")
        choice = input("Enter choice (n/a/t): ").strip().lower()

        collect_new = choice == "n"
        add_new = choice == "a"
        train_only = choice == "t"

        if not (collect_new or add_new or train_only):
            raise SystemExit("Invalid choice. Use n/a/t.")

        samples_per_label_raw = input(
            f"Number of samples (frames) per label [{DEFAULT_SAMPLES_PER_LABEL}]: "
        ).strip()
        samples_per_label = (
            int(samples_per_label_raw) if samples_per_label_raw else DEFAULT_SAMPLES_PER_LABEL
        )
        samples_per_label = max(1, samples_per_label)

        # In interactive mode, motion labels are asked later during collection.
        collect_motion_cli = None
        motion_labels_cli = None
        gestures_cli = None
    else:
        mode = args.mode
        collect_new = mode == "n"
        add_new = mode == "a"
        train_only = mode == "t"
        _ = train_only

        if not (collect_new or add_new or train_only):
            raise SystemExit("Invalid mode. Use n/a/t.")

        samples_per_label = (
            args.samples_per_label if args.samples_per_label is not None else DEFAULT_SAMPLES_PER_LABEL
        )
        samples_per_label = max(1, int(samples_per_label))

        gestures_cli = args.gestures
        collect_motion_cli = args.collect_motion
        if args.motion_labels:
            motion_labels_cli = [x.strip().upper() for x in args.motion_labels.split(",") if x.strip()]
        else:
            motion_labels_cli = None

    # =============================
    # INIT LANDMARKER
    # =============================
    landmarkers = init_landmarker()

    # =============================
    # DATA COLLECTION
    # =============================
    if collect_new or add_new:
        cap, cam_index = open_camera(getattr(args, "camera_index", 0) if args.mode is not None else 0)
        if cap is None:
            raise RuntimeError(
                "Could not open a webcam (tried indices 0-3). "
                "Check permissions to /dev/video*, or try a different camera index."
            )

        print(f"\nCollecting data (camera index {cam_index})...")

        collected = []
        feature_len = None

        if add_new:
            if args.mode is not None:
                if not gestures_cli:
                    raise SystemExit("Mode 'a' requires --gestures for non-interactive training.")
                target_gestures = [x.strip().upper() for x in gestures_cli.split(",") if x.strip()]
            else:
                target_gestures = _parse_label_list(
                    "Enter NEW gesture names (comma separated), e.g. A,B,C or HELLO,YES: ",
                    default_list=None,
                )
            if not target_gestures:
                raise SystemExit("No gestures provided.")
        else:
            target_gestures = list(GESTURES)

        # Motion labels: these are added as normal labels into the same dataset,
        # so the existing classifier can learn to predict them too.
        if args.mode is not None:
            collect_motion = bool(collect_motion_cli)
        else:
            collect_motion = input("Also collect motion labels? (y/n) [y]: ").strip().lower()
            collect_motion = (collect_motion == "" or collect_motion.startswith("y"))

        if collect_motion:
            if args.mode is not None and motion_labels_cli:
                motion_labels = motion_labels_cli
            elif args.mode is not None:
                motion_labels = list(DEFAULT_MOTION_LABELS)
            else:
                motion_labels = _parse_label_list(
                    "Motion labels (comma separated) [SPACE,DELETE] e.g. SPACE,DELETE,MOVEUP: ",
                    default_list=DEFAULT_MOTION_LABELS,
                )
        else:
            motion_labels = []

        # Final collection order: gestures first, then motion labels.
        target_labels = _dedupe_keep_order(list(target_gestures) + list(motion_labels))

        for label in target_labels:
            print(f"\nLabel '{label}' — press 's' to start capturing frames")

            # Wait for 's'
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.putText(
                    frame,
                    f"{label} | Press 's'",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    break

            # 2 second timer (lets user place hand)
            for i in range(2, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.putText(
                    frame,
                    f"Starting in {i}",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 255),
                    3,
                )
                cv2.imshow("Collect", frame)
                cv2.waitKey(1000)

            # Capture fixed number of samples (frames) for this label.
            count, frames = 0, 0
            while count < samples_per_label:
                ret, frame = cap.read()
                if not ret:
                    continue

                frames += 1
                timestamp = int(time.time() * 1000)

    # -------------------------
    # Feature Extraction
    # -------------------------
                features, hand_result, face_result = extract_features(frame, landmarkers, timestamp, True)

    # -------------------------
    # Draw Landmarks
    # -------------------------
                draw_hand_landmarks(frame, hand_result)
                draw_face_landmarks(frame, face_result)

    # -------------------------
    # Emotion Detection
    # -------------------------
                emotion = detect_emotion(face_result)
                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2,
                )

    # -------------------------
    # Hand Status (Debug)
    # -------------------------
                if hand_result is None or not hand_result.hand_landmarks:
                    cv2.putText(
                        frame,
                        "Hand: NOT DETECTED",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Hand: DETECTED",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

    # -------------------------
    # IMPORTANT: Prevent Freeze
    # -------------------------
                if features is None:
                    cv2.imshow("Collect", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

    # -------------------------
    # Feature Validation
    # -------------------------
                if feature_len is None:
                    feature_len = len(features)

                if len(features) != feature_len:
                    continue

    # -------------------------
    # Save Data
    # -------------------------
                collected.append(np.append(features, label))
                count += 1

    # -------------------------
    # UI Text
    # -------------------------
                cv2.putText(
                    frame,
                    f"{label}: {count}/{samples_per_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"All frames: {frames}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow("Collect", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break



                if feature_len is None:
                    feature_len = len(features)

                if len(features) != feature_len:
                    continue

                collected.append(np.append(features, label))
                count += 1

                cv2.putText(
                    frame,
                    f"{label}: {count}/{samples_per_label}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"All frames: {frames}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            print(f"{label} ✓")

        cap.release()
        cv2.destroyAllWindows()

        if len(collected) == 0:
            print("No data collected. Stopping training.")
            raise SystemExit(0)

        # SAFE SAVE: always merge with existing dataset (so training uses old + new).
        new_df = pd.DataFrame(collected)

        # =============================
# SAFE MERGE OLD + NEW DATA
# =============================
        if os.path.exists(DATA_FILE):
            print("Merging with existing dataset...")

            shutil.copy(DATA_FILE, BACKUP_FILE)

            old_df = pd.read_csv(DATA_FILE, header=None, low_memory=False)

            # Ensure same number of columns
            if old_df.shape[1] != new_df.shape[1]:
                print("⚠ Feature mismatch! Skipping old data.")
            else:
                new_df = pd.concat([old_df, new_df], ignore_index=True)
                print(f"Old samples: {len(old_df)} | New samples: {len(collected)}")
                print(f"Total samples after merge: {len(new_df)}")

# Save merged dataset
        new_df.to_csv(DATA_FILE, index=False, header=False)
        print("Dataset saved successfully ✔")

    # =============================
    # TRAIN MODEL (always)
    # =============================
    if not os.path.exists(DATA_FILE):
        raise SystemExit("No dataset found. Collect data first (run with 'n' or 'a').")

    df = pd.read_csv(DATA_FILE, header=None, low_memory=False)

    labels = df.iloc[:, -1]
    features = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")

    # Remove corrupted rows (where conversion failed)
    valid = features.notna().all(axis=1)
    X = features[valid].astype(float)
    y = labels[valid].astype(str)

    print(f"Valid samples: {len(X)}")
    if len(X) < 10:
        raise SystemExit("Not enough valid data to train.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    joblib.dump(model, MODEL_FILE)
    print("Model + Scaler saved ✔")


if __name__ == "__main__":
    main()

