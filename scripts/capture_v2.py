"""Phase 1 Step 4: Capture your own ISL alphabet hold-out set (v2, raw landmarks).

Press a letter key (A-Z) to save the current frame's RAW landmarks
(raw[2,21,3], present[2]) plus the letter and a timestamp to
data/holdout/own_capture_v2.npz. Press ESC to quit. Press D to delete the last
saved sample. Shows MediaPipe landmarks on screen and per-letter counts.
Saves landmarks only, never images.

Runs MediaPipe on the UNFLIPPED camera frame; flips only the displayed image.

Run: .venv-mp/Scripts/python scripts/capture_v2.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_PATH = REPO / "models" / "hand_landmarker.task"
HOLD_DIR = REPO / "data" / "holdout"
NPZ_PATH = HOLD_DIR / "own_capture_v2.npz"
COUNTS_PATH = HOLD_DIR / "counts_v2.json"

CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
RAW_SHAPE = (2, 21, 3)


def build_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_raw(frame_bgr, detector):
    """Return (raw[2,21,3], present[2], handed_label[2], handed_score[2], annotated_frame)."""
    raw = np.zeros(RAW_SHAPE, dtype=np.float32)
    present = np.zeros(2, dtype=bool)
    handed_label = ["", ""]
    handed_score = np.zeros(2, dtype=np.float32)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    hands = result.hand_landmarks
    handedness = result.handedness

    if not hands:
        return raw, present, handed_label, handed_score, frame_bgr

    # Order by wrist x ascending (slot 0 = leftmost in image)
    hand_info = []
    for h_lm, h_hand in zip(hands, handedness):
        wrist_x = h_lm[0].x
        label = h_hand[0].category_name
        score = h_hand[0].score
        hand_info.append((wrist_x, h_lm, label, score))
    hand_info.sort(key=lambda t: t[0])

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    for slot, (wrist_x, h_lm, label, score) in enumerate(hand_info):
        if slot >= 2:
            break
        pts = np.array([[lm.x, lm.y, lm.z] for lm in h_lm], dtype=np.float32)
        raw[slot] = pts
        present[slot] = True
        handed_label[slot] = label
        handed_score[slot] = score
        # Draw landmarks
        for lm in h_lm:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 0), -1)

    # Flip only the displayed image
    annotated = cv2.flip(annotated, 1)
    return raw, present, handed_label, handed_score, annotated


def load_existing():
    data = {"raw": [], "present": [], "label": [], "ts": []}
    counts = {c: 0 for c in CLASSES}
    if NPZ_PATH.exists():
        with np.load(NPZ_PATH, allow_pickle=True) as z:
            data["raw"] = z["raw"].tolist()
            data["present"] = z["present"].tolist()
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
        raw=np.array(data["raw"], dtype=np.float32),
        present=np.array(data["present"], dtype=bool),
        label=np.array(data["label"], dtype=object),
        ts=np.array(data["ts"], dtype=object),
    )
    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def main():
    if not MODEL_PATH.exists():
        print(f"FATAL: model file not found: {MODEL_PATH}")
        sys.exit(1)
    detector = build_detector()
    data, counts = load_existing()
    total_saved = len(data["raw"])
    print(f"Loaded {total_saved} existing captures. Counts: {counts}")
    print("Press letter A-Z to save. D to delete last. ESC to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FATAL: could not open webcam")
        sys.exit(1)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("FATAL: could not read frame")
                break
            raw, present, handed_label, handed_score, annotated = extract_raw(frame, detector)

            # HUD (on flipped display image)
            cv2.putText(annotated, "ISL Hold-out Capture v2 - press A-Z, D=delete, ESC=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated, f"Total saved: {total_saved}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            nh = int(present.sum())
            cv2.putText(annotated, f"Hands: {nh}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if nh else (0, 0, 255), 1)
            cnt_str = " ".join(f"{c}:{counts[c]}" for c in CLASSES)
            cv2.putText(annotated, cnt_str, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if nh == 0:
                cv2.putText(annotated, "No hand detected", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            cv2.imshow("SignBridge Hold-out Capture v2", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord("d") or key == ord("D"):
                if total_saved > 0:
                    data["raw"].pop()
                    data["present"].pop()
                    last_label = data["label"].pop()
                    data["ts"].pop()
                    counts[last_label] = max(0, counts.get(last_label, 0) - 1)
                    total_saved -= 1
                    print(f"Deleted last '{last_label}'. Total: {total_saved}")
                    save_all(data, counts)
            elif key != 255:
                ch = chr(key).upper()
                if ch in CLASSES:
                    if nh == 0:
                        print(f"No hand detected; not saving '{ch}'")
                        continue
                    ts = datetime.now().isoformat(timespec="seconds")
                    data["raw"].append(raw)
                    data["present"].append(present)
                    data["label"].append(ch)
                    data["ts"].append(ts)
                    counts[ch] = counts.get(ch, 0) + 1
                    total_saved += 1
                    print(f"Saved '{ch}' (#{counts[ch]}). Total: {total_saved}")
                    save_all(data, counts)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_all(data, counts)
        print(f"Done. Saved {total_saved} captures to {NPZ_PATH}")
        print("Counts:", counts)


if __name__ == "__main__":
    main()
