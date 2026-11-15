"""Phase 1 Step 1: Re-extract landmarks with RAW coordinates (no handedness slot bias).

Reads data/landmarks/metadata.csv (rel_path, split, cls, sha256, phash) - does NOT recompute hashes.
Processes images in the same row order as metadata.csv.
Uses MediaPipe Tasks HandLandmarker (models/hand_landmarker.task, IMAGE mode).
Stores RAW landmarks in image-normalized coordinates: raw[2,21,3] float32, present[2] bool.
Orders detected hands by wrist (landmark 0) x ascending so slot 0 is leftmost in image.
1 hand goes in slot 0; slot 1 zeros.
Stores handedness label and score per slot as metadata ONLY (never used as feature).
Saves chunks as data/landmarks/v2_parts/part_XXXX.npz (1000 images each), then merges into
data/landmarks/alphabet_v2.npz with arrays: raw, present, handed_label, handed_score,
rel_path, split, cls (same order as metadata.csv). Resume = skip parts that exist.

Run: .venv-mp/Scripts/python signbridge/extract_v2.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

REPO = Path(__file__).resolve().parents[1]
META_PATH = REPO / "data" / "landmarks" / "metadata.csv"
MODEL_PATH = REPO / "models" / "hand_landmarker.task"
PARTS_DIR = REPO / "data" / "landmarks" / "v2_parts"
MERGED_PATH = REPO / "data" / "landmarks" / "alphabet_v2.npz"
LOG_DIR = REPO / "reports" / "phase1" / "logs"
LOG_PATH = LOG_DIR / "extract_v2.log"

CHUNK_SIZE = 1000
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
RAW_SHAPE = (NUM_HANDS, LANDMARKS_PER_HAND, 3)


def log(msg):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def load_metadata():
    rows = []
    with open(META_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def part_path(index):
    return PARTS_DIR / f"part_{index:04d}.npz"


def part_exists(index):
    return part_path(index).exists()


def save_part(index, records):
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    p = part_path(index)
    raw_arr = np.stack([r["raw"] for r in records]).astype(np.float32)
    present_arr = np.stack([r["present"] for r in records]).astype(np.bool_)
    handed_label_arr = np.array([r["handed_label"] for r in records], dtype=object)
    handed_score_arr = np.stack([r["handed_score"] for r in records]).astype(np.float32)
    rel_path_arr = np.array([r["rel_path"] for r in records], dtype=object)
    split_arr = np.array([r["split"] for r in records], dtype=object)
    cls_arr = np.array([r["cls"] for r in records], dtype=object)
    np.savez_compressed(
        p,
        raw=raw_arr,
        present=present_arr,
        handed_label=handed_label_arr,
        handed_score=handed_score_arr,
        rel_path=rel_path_arr,
        split=split_arr,
        cls=cls_arr,
    )


def build_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_from_image(rgb_arr, detector):
    """Return (raw[2,21,3], present[2], handed_label[2], handed_score[2])."""
    raw = np.zeros(RAW_SHAPE, dtype=np.float32)
    present = np.zeros(NUM_HANDS, dtype=np.bool_)
    handed_label = ["", ""]
    handed_score = np.zeros(NUM_HANDS, dtype=np.float32)

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr)
    result = detector.detect(mp_img)
    hands = result.hand_landmarks
    handedness = result.handedness

    if not hands:
        return raw, present, handed_label, handed_score

    # Build list of (wrist_x, landmarks, handedness)
    hand_info = []
    for h_lm, h_hand in zip(hands, handedness):
        lm = h_lm[0]  # first landmark is wrist
        wrist_x = lm.x
        label = h_hand[0].category_name
        score = h_hand[0].score
        hand_info.append((wrist_x, h_lm, label, score))

    # Sort by wrist x ascending -> slot 0 = leftmost in image
    hand_info.sort(key=lambda t: t[0])

    for slot, (wrist_x, h_lm, label, score) in enumerate(hand_info):
        if slot >= NUM_HANDS:
            break
        pts = np.array([[lm.x, lm.y, lm.z] for lm in h_lm], dtype=np.float32)
        raw[slot] = pts
        present[slot] = True
        handed_label[slot] = label
        handed_score[slot] = score

    return raw, present, handed_label, handed_score


def merge_parts(total_records):
    """Merge all part files into alphabet_v2.npz in metadata order."""
    log("Merging parts into alphabet_v2.npz ...")
    part_indices = sorted(
        int(p.stem.split("_")[1]) for p in PARTS_DIR.glob("part_*.npz")
    )
    # Build index by rel_path
    by_path = {}
    for idx in part_indices:
        with np.load(part_path(idx), allow_pickle=True) as z:
            raw_arr = z["raw"]
            present_arr = z["present"]
            handed_label_arr = z["handed_label"]
            handed_score_arr = z["handed_score"]
            rel_path_arr = z["rel_path"]
            split_arr = z["split"]
            cls_arr = z["cls"]
            for i in range(len(rel_path_arr)):
                rp = rel_path_arr[i]
                by_path[rp] = {
                    "raw": raw_arr[i],
                    "present": present_arr[i],
                    "handed_label": handed_label_arr[i],
                    "handed_score": handed_score_arr[i],
                    "rel_path": rp,
                    "split": split_arr[i],
                    "cls": cls_arr[i],
                }

    # Order by metadata.csv
    rows = load_metadata()
    raw_list, present_list = [], []
    handed_label_list, handed_score_list = [], []
    rel_path_list, split_list, cls_list = [], [], []
    missing = 0
    for r in rows:
        rp = r["rel_path"]
        if rp in by_path:
            d = by_path[rp]
            raw_list.append(d["raw"])
            present_list.append(d["present"])
            handed_label_list.append(d["handed_label"])
            handed_score_list.append(d["handed_score"])
            rel_path_list.append(d["rel_path"])
            split_list.append(d["split"])
            cls_list.append(d["cls"])
        else:
            missing += 1

    raw_arr = np.stack(raw_list).astype(np.float32)
    present_arr = np.stack(present_list).astype(np.bool_)
    handed_label_arr = np.array(handed_label_list, dtype=object)
    handed_score_arr = np.stack(handed_score_list).astype(np.float32)
    rel_path_arr = np.array(rel_path_list, dtype=object)
    split_arr = np.array(split_list, dtype=object)
    cls_arr = np.array(cls_list, dtype=object)

    np.savez_compressed(
        MERGED_PATH,
        raw=raw_arr,
        present=present_arr,
        handed_label=handed_label_arr,
        handed_score=handed_score_arr,
        rel_path=rel_path_arr,
        split=split_arr,
        cls=cls_arr,
    )
    log(f"Merged {len(raw_list)} records ({missing} missing) into {MERGED_PATH}")


def main():
    if not MODEL_PATH.exists():
        log(f"FATAL: model file not found: {MODEL_PATH}")
        sys.exit(1)
    if not META_PATH.exists():
        log(f"FATAL: metadata not found: {META_PATH}")
        sys.exit(1)

    detector = build_detector()
    log("Detector created. Loading metadata...")

    rows = load_metadata()
    log(f"Total images in metadata: {len(rows)}")

    # Determine resume point: skip parts that exist
    existing_parts = set()
    if PARTS_DIR.exists():
        for p in PARTS_DIR.glob("part_*.npz"):
            try:
                idx = int(p.stem.split("_")[1])
                existing_parts.add(idx)
            except ValueError:
                pass

    next_part = 0
    if existing_parts:
        next_part = max(existing_parts) + 1
    log(f"Existing parts: {sorted(existing_parts)}. Next part to write: {next_part}")

    records = []
    records_in_part = 0
    part_index = next_part
    total_processed = 0
    total_skipped = 0
    detection_counter = {"0": 0, "1": 0, "2": 0}
    same_label_2hand = 0
    total_2hand = 0
    per_class_detect = {}
    t0 = time.time()

    for row_index, r in enumerate(rows):
        rp = r["rel_path"]
        split = r["split"]
        cls = r["cls"]

        # Determine which part this row belongs to
        row_part = row_index // CHUNK_SIZE

        # If this part already exists, skip all rows in it
        if row_part in existing_parts:
            total_skipped += 1
            # Still need to count detection stats? No - already counted before.
            continue

        # We are in a new part (not yet written). Process this row.
        # Load image
        img_path = REPO / rp
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                arr = np.array(im.convert("RGB"))
        except Exception as e:
            log(f"READ_ERROR {rp}: {e}")
            continue

        raw, present, handed_label, handed_score = extract_from_image(arr, detector)
        nh = int(present.sum())
        detection_counter[str(nh)] = detection_counter.get(str(nh), 0) + 1

        if nh == 2:
            total_2hand += 1
            if handed_label[0] == handed_label[1] and handed_label[0] != "":
                same_label_2hand += 1

        per_class_detect.setdefault(cls, {"0": 0, "1": 0, "2": 0})
        per_class_detect[cls][str(nh)] += 1

        records.append({
            "raw": raw,
            "present": present,
            "handed_label": handed_label,
            "handed_score": handed_score,
            "rel_path": rp,
            "split": split,
            "cls": cls,
        })
        records_in_part += 1
        total_processed += 1

        # Write part when full
        if records_in_part >= CHUNK_SIZE:
            save_part(part_index, records)
            log(f"Saved part {part_index} ({len(records)} records). "
                f"Processed {total_processed} new images, skipped {total_skipped}. "
                f"Detection so far: {detection_counter}. Same-label 2-hand: {same_label_2hand}/{total_2hand}")
            records = []
            records_in_part = 0
            part_index += 1

    # Write final partial part
    if records:
        save_part(part_index, records)
        log(f"Saved final part {part_index} ({len(records)} records).")

    log(f"Extraction complete. New images processed: {total_processed}, skipped (already in parts): {total_skipped}")
    log(f"Detection counts: {detection_counter}")
    log(f"2-hand images: {total_2hand}, same-label count: {same_label_2hand}")

    # Per-class detection summary
    log("Per-class detection (num_hands>0):")
    for cls in sorted(per_class_detect.keys()):
        d = per_class_detect[cls]
        total_c = sum(d.values())
        detect_c = d["1"] + d["2"]
        log(f"  {cls}: {detect_c}/{total_c} = {detect_c/total_c*100:.1f}% detected")

    # Merge parts
    merge_parts(total_processed + total_skipped)

    elapsed = time.time() - t0
    log(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
