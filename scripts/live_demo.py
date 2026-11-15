"""Phase 1 Step 5: Live webcam demo predicting the ISL letter with the saved model.

Webcam loop: predicts the letter live and shows top-3 probabilities.
DO NOT RUN during this session; only compile it.

Run:  .venv-mp/Scripts/python scripts/live_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = REPO / "models" / "landmark_baseline.joblib"
LANDMARK_MODEL = REPO / "models" / "hand_landmarker.task"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
WRIST_IDX, MIDDLE_MCP_IDX = 0, 9


def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    wrist = pts[WRIST_IDX]
    mid_mcp = pts[MIDDLE_MCP_IDX]
    scale = float(np.linalg.norm(mid_mcp - wrist))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    return ((pts - wrist) / scale).reshape(-1).astype(np.float32)


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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            letter, probs = None, None
            if result.hand_landmarks:
                hands = result.hand_landmarks
                handedness = result.handedness
                slot_vecs = [np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)]
                for h_lm, h_hand in zip(hands, handedness):
                    label = h_hand[0].category_name
                    slot = 0 if label == "Left" else 1
                    slot_vecs[slot] = normalize_hand(h_lm)
                feat = np.concatenate(slot_vecs)[None, ...]
                probs = clf.predict_proba(feat)[0]
                top3 = probs.argsort()[::-1][:3]
                letter = CLASSES[int(probs.argmax())]

            h, w = frame.shape[:2]
            cv2.putText(frame, "ISL Live Demo - ESC to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if letter:
                cv2.putText(frame, f"Prediction: {letter}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                for rank, idx in enumerate(top3):
                    cv2.putText(frame, f"{rank+1}. {CLASSES[idx]} {probs[idx]:.2f}",
                                (10, 110 + rank * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("SignBridge Live Demo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
