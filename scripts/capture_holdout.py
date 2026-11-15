"""Phase 1 Step 5: Capture your own ISL alphabet hold-out set from the webcam.

Press a letter key (A-Z) to save the current frame's 126-feature landmark vector
(plus the letter and a timestamp) to data/holdout/own_capture.npz. Press ESC to quit.
Shows MediaPipe landmarks on screen and the count saved per letter.
Saves landmarks only, never images.

Run:  .venv-mp/Scripts/python scripts/capture_holdout.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import mediapipe as mp

MODEL_PATH = REPO / "models" / "hand_landmarker.task"
HOLD_DIR = REPO / "data" / "holdout"
NPZ_PATH = HOLD_DIR / "own_capture.npz"
COUNTS_PATH = HOLD_DIR / "counts.json"

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


def build_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_features(mp_img, detector):
    """Return (feat126, num_hands, landmarks_list) or (None, 0, []) if no hand."""
    result = detector.detect(mp_img)
    hands = result.hand_landmarks
    if not hands:
        return None, 0, []
    slot_vecs = [np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)]
    for h_lm, h_hand in zip(hands, result.handedness):
        label = h_hand[0].category_name
        slot = 0 if label == "Left" else 1
        slot_vecs[slot] = normalize_hand(h_lm)
    feat = np.concatenate(slot_vecs)
    return feat, len(hands), hands


def load_existing():
    data = {"feat": [], "label": [], "ts": []}
    counts = {c: 0 for c in CLASSES}
    if NPZ_PATH.exists():
        with np.load(NPZ_PATH, allow_pickle=True) as z:
            data["feat"] = z["feat"].tolist()
            data["label"] = z["label"].tolist()
            data["ts"] = z["ts"].tolist()
        for lbl in data["label"]:
            if lbl in counts:
                counts[lbl] += 1
    if COUNTS_PATH.exists():
        with open(COUNTS_PATH, "r", encoding="utf-8") as f:
            counts = json.load(f)
    return data, counts


def save_all(data, counts):
    HOLD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ_PATH,
        feat=np.array(data["feat"], dtype=np.float32),
        label=np.array(data["label"], dtype=object),
        ts=np.array(data["ts"], dtype=object),
    )
    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def draw_landmarks(frame, hands):
    h, w = frame.shape[:2]
    connections = getattr(vision, "HandLandmarksConnections", None)
    for h_lm in hands:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in h_lm]
        for (x, y) in pts:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        if connections:
            for a, b in connections:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)


def main():
    if not MODEL_PATH.exists():
        print(f"FATAL: model file not found: {MODEL_PATH}")
        print("Run extract.py once (it downloads hand_landmarker.task) or place the .task file.")
        sys.exit(1)
    detector = build_detector()
    data, counts = load_existing()
    total_saved = len(data["feat"])
    print(f"Loaded {total_saved} existing captures. Counts: {counts}")
    print("Press a letter key (A-Z) to save. ESC to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL: could not open webcam (cv2.VideoCapture(0))")
        sys.exit(1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("FATAL: could not read frame from webcam")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            feat, num_hands, hands = extract_features(mp_img, detector)

            draw_landmarks(frame, hands)

            # HUD
            cv2.putText(frame, "ISL Hold-out Capture - press letter A-Z, ESC to quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Total saved: {total_saved}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Hands: {num_hands}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if num_hands else (0, 0, 255), 1)
            # per-letter counts line
            cnt_str = " ".join(f"{c}:{counts[c]}" for c in CLASSES)
            cv2.putText(frame, cnt_str, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if feat is None:
                cv2.putText(frame, "No hand detected", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("SignBridge Hold-out Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key != 255:  # a key was pressed
                ch = chr(key).upper()
                if ch in CLASSES:
                    if feat is None:
                        print(f"No hand detected; not saving '{ch}'")
                        continue
                    ts = datetime.now().isoformat(timespec="seconds")
                    data["feat"].append(feat)
                    data["label"].append(ch)
                    data["ts"].append(ts)
                    counts[ch] = counts.get(ch, 0) + 1
                    total_saved += 1
                    print(f"Saved '{ch}' (# {counts[ch]}). Total: {total_saved}")
                    save_all(data, counts)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_all(data, counts)
        print(f"Done. Saved {total_saved} captures to {NPZ_PATH}")
        print("Counts:", counts)


if __name__ == "__main__":
    main()
