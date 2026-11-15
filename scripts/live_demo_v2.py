"""Phase 1 Step 4: Live webcam demo with v2 slot-invariant model.

Webcam loop: predicts the letter live, shows top-3 probabilities, smooths with a
5-frame majority vote, shows "No hand" when none detected.
Runs MediaPipe on the UNFLIPPED camera frame; flips only the displayed image.

DO NOT RUN during this session; only python -m py_compile it.

Run: .venv-mp/Scripts/python scripts/live_demo_v2.py
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from signbridge.features import build_features, FEATURE_DIM  # noqa: E402

MODEL_PATH = REPO / "models" / "landmark_v2.joblib"
LANDMARK_MODEL = REPO / "models" / "hand_landmarker.task"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
SMOOTH_WINDOW = 5


def main():
    if not MODEL_PATH.exists():
        print(f"FATAL: model not found: {MODEL_PATH}")
        sys.exit(1)
    if not LANDMARK_MODEL.exists():
        print(f"FATAL: landmark model not found: {LANDMARK_MODEL}")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    base_options = mp_tasks.BaseOptions(model_asset_path=str(LANDMARK_MODEL))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL: could not open webcam")
        sys.exit(1)

    vote_buf = deque(maxlen=SMOOTH_WINDOW)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # flip for display only

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            letter, probs = None, None
            if result.hand_landmarks:
                hands = result.hand_landmarks
                handedness = result.handedness
                # Build raw[2,21,3], present[2] ordered by wrist x ascending
                raw = np.zeros((2, 21, 3), dtype=np.float32)
                present = np.zeros(2, dtype=bool)
                hand_info = []
                for h_lm, h_hand in zip(hands, handedness):
                    hand_info.append((h_lm[0].x, h_lm, h_hand))
                hand_info.sort(key=lambda t: t[0])
                for slot, (_, h_lm, _) in enumerate(hand_info):
                    if slot >= 2:
                        break
                    raw[slot] = np.array([[lm.x, lm.y, lm.z] for lm in h_lm], dtype=np.float32)
                    present[slot] = True

                feat = build_features(raw, present, mirror=False)[None, ...]
                probs = clf.predict_proba(feat)[0]
                letter = CLASSES[int(probs.argmax())]
                vote_buf.append(letter)
            else:
                vote_buf.append(None)

            # Smoothed letter
            votes = [v for v in vote_buf if v is not None]
            smoothed = max(set(votes), key=votes.count) if votes else None

            h, w = frame.shape[:2]
            cv2.putText(frame, "ISL Live Demo v2 - ESC to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if smoothed:
                cv2.putText(frame, f"Prediction: {smoothed}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                if probs is not None:
                    top3 = probs.argsort()[::-1][:3]
                    for rank, idx in enumerate(top3):
                        cv2.putText(frame, f"{rank+1}. {CLASSES[idx]} {probs[idx]:.2f}",
                                    (10, 110 + rank * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("SignBridge Live Demo v2", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
