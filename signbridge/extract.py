"""Phase 1 Step 2: Extract MediaPipe hand landmarks from ISL alphabet images.

Read-only input: datasets/processed/RealSign/Alphabet/{Training,Validation,Testing}/<A-Z>/*.jpg
Outputs:  data/landmarks/alphabet.npz   (feature vectors + metadata arrays)
          data/landmarks/metadata.csv   (one row per image)
          data/landmarks/progress.json  (resumability checkpoint)
          reports/phase1/logs/extract.log (no-hand detections + progress)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "datasets" / "processed" / "RealSign" / "Alphabet"
OUT_DIR = REPO / "data" / "landmarks"
LOG_PATH = REPO / "reports" / "phase1" / "logs" / "extract.log"
MODEL_PATH = REPO / "models" / "hand_landmarker.task"

NPZ_PATH = OUT_DIR / "alphabet.npz"
CSV_PATH = OUT_DIR / "metadata.csv"
PROGRESS_PATH = OUT_DIR / "progress.json"

CHUNK_SIZE = 1000
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEAT_PER_HAND = LANDMARKS_PER_HAND * 3  # x,y,z
TOTAL_FEAT = NUM_HANDS * FEAT_PER_HAND  # 126

SPLITS = ["Training", "Validation", "Testing"]
CLASS_FOLDERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_image(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB")))


def normalize_hand(landmarks) -> np.ndarray:
    """Return 63-vector: (xyz - wrist) / wrist->middle-MCP distance, flattened."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    wrist = pts[WRIST_IDX]
    mid_mcp = pts[MIDDLE_MCP_IDX]
    scale = float(np.linalg.norm(mid_mcp - wrist))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    norm = (pts - wrist) / scale
    return norm.reshape(-1)


def build_detector():
    base_options = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def list_images():
    """Yield (rel_path, split, class_label, abs_path) for every image."""
    for split in SPLITS:
        split_dir = DATA_ROOT / split
        if not split_dir.is_dir():
            continue
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            cls = class_folder.name
            if cls not in CLASS_FOLDERS:
                continue
            for img_path in sorted(class_folder.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                rel = img_path.relative_to(REPO)
                yield str(rel), split, cls, img_path


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": [], "n_total": 0}


def save_progress(done):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump({"done": done, "n_total": len(done)}, f)


def append_chunk(records, features, no_hand_list):
    """Write all accumulated records to NPZ + CSV (rewrite each chunk)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(records)
    X = np.zeros((n, TOTAL_FEAT), dtype=np.float32)
    meta = []
    for i, rec in enumerate(records):
        X[i] = rec["features"]
        meta.append(rec)

    np.savez_compressed(
        NPZ_PATH,
        features=X,
        rel_path=np.array([m["rel_path"] for m in meta], dtype=object),
        split=np.array([m["split"] for m in meta], dtype=object),
        cls=np.array([m["cls"] for m in meta], dtype=object),
        sha256=np.array([m["sha256"] for m in meta], dtype=object),
        phash=np.array([m["phash"] for m in meta], dtype=object),
        num_hands=np.array([m["num_hands"] for m in meta], dtype=np.int8),
    )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["rel_path", "split", "cls", "sha256", "phash", "num_hands"]
        )
        for m in meta:
            w.writerow(
                [m["rel_path"], m["split"], m["cls"], m["sha256"], m["phash"], m["num_hands"]]
            )

    # Append no-hand log
    if no_hand_list:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            for nh in no_hand_list:
                f.write("NO_HAND\t" + nh["rel_path"] + "\t" + nh["split"] + "\t" + nh["cls"] + "\n")


def main():
    if not MODEL_PATH.exists():
        log(f"FATAL: model file not found: {MODEL_PATH}")
        sys.exit(1)

    detector = build_detector()
    log("Detector created. Scanning images...")

    all_images = list(list_images())
    log(f"Total images found: {len(all_images)}")

    progress = load_progress()
    done_set = set(progress.get("done", []))

    records = []
    features_list = []
    no_hand_buffer = []
    resume_count = 0

    # Reload existing NPZ/CSV if resuming
    start_idx = 0
    if NPZ_PATH.exists() and done_set:
        with np.load(NPZ_PATH, allow_pickle=True) as z:
            existing_paths = z["rel_path"].tolist()
            existing_features = z["features"].tolist()
        for p, feat in zip(existing_paths, existing_features):
            records.append({"rel_path": p, "features": np.asarray(feat, dtype=np.float32)})
            features_list.append(np.asarray(feat, dtype=np.float32))
        resume_count = len(records)
        log(f"Resuming: {resume_count} records already extracted.")

    t0 = time.time()
    processed_since_chunk = 0
    no_hand_total = 0

    for idx, (rel_path, split, cls, img_path) in enumerate(all_images):
        if rel_path in done_set:
            continue

        try:
            with Image.open(img_path) as im:
                arr = np.array(im.convert("RGB"))
        except Exception as e:
            log(f"READ_ERROR\t{rel_path}\t{split}\t{cls}\t{e}")
            continue

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        result = detector.detect(mp_img)

        hands = result.hand_landmarks
        handedness = result.handedness
        num_hands = len(hands)

        if num_hands == 0:
            no_hand_total += 1
            no_hand_buffer.append({"rel_path": rel_path, "split": split, "cls": cls})
            feat = np.zeros(TOTAL_FEAT, dtype=np.float32)
        else:
            # Build slot vectors: slot 0 = Left, slot 1 = Right
            slot_vecs = [np.zeros(FEAT_PER_HAND, dtype=np.float32),
                         np.zeros(FEAT_PER_HAND, dtype=np.float32)]
            for h_lm, h_hand in zip(hands, handedness):
                label = h_hand[0].category_name if hasattr(h_hand, "__getitem__") else h_hand.category_name
                slot = 0 if label == "Left" else 1
                slot_vecs[slot] = normalize_hand(h_lm).astype(np.float32)
            feat = np.concatenate([slot_vecs[0], slot_vecs[1]])

        records.append({"rel_path": rel_path, "features": feat})
        features_list.append(feat)

        # Metadata is reconstructed at save time from records + parallel arrays
        # We store sha256/phash at save time to avoid recomputing on resume.
        # Compute sha256 + phash now (cheap, file bytes only).
        sha = sha256_file(img_path)
        ph = phash_image(img_path)
        records[-1]["sha256"] = sha
        records[-1]["phash"] = ph
        records[-1]["split"] = split
        records[-1]["cls"] = cls
        records[-1]["num_hands"] = num_hands

        done_set.add(rel_path)
        processed_since_chunk += 1

        if processed_since_chunk >= CHUNK_SIZE:
            # Rebuild full meta arrays for save
            append_chunk(records, features_list, no_hand_buffer)
            no_hand_buffer = []
            save_progress(sorted(done_set))
            elapsed = time.time() - t0
            log(f"Chunk saved: {len(records)} records, {no_hand_total} no-hand so far, "
                f"{elapsed:.1f}s elapsed")
            processed_since_chunk = 0

    # Final save
    append_chunk(records, features_list, no_hand_buffer)
    save_progress(sorted(done_set))

    elapsed = time.time() - t0
    log(f"DONE: {len(records)} images extracted in {elapsed:.1f}s, "
        f"{no_hand_total} with no hand detected.")

    # Detection rate summary
    from collections import Counter, defaultdict
    det_count = Counter()
    total_count = Counter()
    det_split = Counter()
    total_split = Counter()
    for m in records:
        total_count[m["cls"]] += 1
        total_split[m["split"]] += 1
        if m["num_hands"] > 0:
            det_count[m["cls"]] += 1
            det_split[m["split"]] += 1

    log("\n--- Detection rate by class ---")
    class_rates = []
    for cls in CLASS_FOLDERS:
        tot = total_count.get(cls, 0)
        det = det_count.get(cls, 0)
        rate = det / tot * 100 if tot else 0.0
        class_rates.append((cls, rate, det, tot))
        log(f"{cls}: {det}/{tot} = {rate:.1f}%")

    class_rates.sort(key=lambda x: x[1])
    log("5 worst classes by detection rate:")
    for cls, rate, det, tot in class_rates[:5]:
        log(f"  {cls}: {det}/{tot} = {rate:.1f}%")

    log("\n--- Detection rate by split ---")
    for split in SPLITS:
        tot = total_split.get(split, 0)
        det = det_split.get(split, 0)
        rate = det / tot * 100 if tot else 0.0
        log(f"{split}: {det}/{tot} = {rate:.1f}%")

    overall_det = sum(det_count.values())
    overall_tot = sum(total_count.values())
    log(f"\nOVERALL: {overall_det}/{overall_tot} = {overall_det/overall_tot*100:.1f}% detected")
    log(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
