"""Phase 1 slot-invariant baseline (NOT yet runnable — see note below).

Builds handedness-label-independent features and trains a RandomForest
with slot-swap + mirror augmentation on the new train split.

BLOCKER: raw wrist x-coordinate is NOT stored in data/landmarks/alphabet.npz.
The extractor normalizes each hand by translating the wrist to the origin,
so absolute image x-position is lost. Ordering 2-hand images by wrist x
requires re-extraction with an extra raw-wrist-x field. Do NOT re-run the
31-minute extraction without being asked.

To make this runnable: re-extract landmarks storing, per hand slot,
an additional raw_wrist_x value (and raw_wrist_y if desired) alongside the
126-dim feature vector, then update the FEATURE_DIM / WRIST_X_INDEX below.

Run:  .venv-mp/Scripts/python scripts/train_slot_invariant.py
"""
from __future__ import annotations

import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "data" / "landmarks" / "split_assignments.csv"
NPZ_PATH = REPO / "data" / "landmarks" / "alphabet.npz"
MODEL_PATH = REPO / "models" / "landmark_invariant.joblib"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# ---------------------------------------------------------------------------
# CONFIGURATION (update after re-extraction adds raw wrist x)
# ---------------------------------------------------------------------------
# Current alphabet.npz stores 126-dim features only (no raw wrist x).
WRIST_X_AVAILABLE = False  # set True once npz includes raw wrist x
FEATURE_DIM = 126
HAND_DIM = 63  # per hand: 21 landmarks * 3

# ---------------------------------------------------------------------------
def load_data():
    assign = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assign[r["rel_path"]] = r
    z = np.load(NPZ_PATH, allow_pickle=True)
    rel_paths = z["rel_path"].tolist()
    features = z["features"].tolist()
    num_hands = z["num_hands"].tolist()
    feat_by_path = {p: np.asarray(ft, dtype=np.float32) for p, ft in zip(rel_paths, features)}
    nh_by_path = {p: int(nh) for p, nh in zip(rel_paths, num_hands)}
    return assign, feat_by_path, nh_by_path, rel_paths

def make_slot_invariant(feat, nh, raw_wrist_x=None):
    """Return a slot-invariant 126-vector.

    1-hand: always place the hand in slot 0, zeros in slot 1.
    2-hand: order hands by raw wrist x (left-to-right in image).
            Falls back to slot-swap augmentation if raw_wrist_x unavailable.
    """
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    if nh == 0:
        return out
    s0 = feat[:HAND_DIM]
    s1 = feat[HAND_DIM:]
    s0_nonzero = np.abs(s0).sum() > 0
    s1_nonzero = np.abs(s1).sum() > 0

    if nh == 1:
        hand = s0 if s0_nonzero else s1
        out[:HAND_DIM] = hand  # always slot 0
        return out

    # nh == 2: both hands present
    if WRIST_X_AVAILABLE and raw_wrist_x is not None:
        wx0, wx1 = raw_wrist_x
        if wx0 <= wx1:
            out[:HAND_DIM] = s0
            out[HAND_DIM:] = s1
        else:
            out[:HAND_DIM] = s1
            out[HAND_DIM:] = s0
    else:
        # Without raw wrist x, cannot deterministically order by image position.
        # Keep as-is; augmentation at train time handles slot ambiguity.
        out[:HAND_DIM] = s0
        out[HAND_DIM:] = s1
    return out

def augment(X, y):
    """Original + slot-swapped + x-negated (mirror) copies."""
    Xs, ys = [X], [y]
    # slot-swapped
    Xswap = X.copy()
    Xswap[:, :HAND_DIM], Xswap[:, HAND_DIM:] = Xswap[:, HAND_DIM:].copy(), Xswap[:, :HAND_DIM].copy()
    Xs.append(Xswap); ys.append(y)
    # mirror (negate x coords)
    Xmir = X.copy()
    x_idx = np.arange(0, FEATURE_DIM, 3)
    Xmir[:, x_idx] = -Xmir[:, x_idx]
    Xs.append(Xmir); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)

def build_arrays(assign, feat_by_path, nh_by_path, rel_paths, new_split):
    Xs, ys = [], []
    for rp in rel_paths:
        if assign[rp]["new_split"] != new_split:
            continue
        if nh_by_path[rp] == 0:
            continue
        Xs.append(make_slot_invariant(feat_by_path[rp], nh_by_path[rp]))
        ys.append(assign[rp]["cls"])
    return (np.stack(Xs) if Xs else np.zeros((0, FEATURE_DIM)),
            np.array(ys))

def main():
    if not WRIST_X_AVAILABLE:
        print("BLOCKER: raw wrist x-coordinate is not stored in alphabet.npz.")
        print("  The extractor normalizes each hand to wrist-origin, losing absolute x.")
        print("  Re-extraction with an extra raw_wrist_x field is required before")
        print("  this script can train a truly slot-invariant model.")
        print("  Script written to scripts/ but NOT run. See WRIST_X_AVAILABLE flag.")
        return

    assign, feat_by_path, nh_by_path, rel_paths = load_data()

    X_tr, y_tr = build_arrays(assign, feat_by_path, nh_by_path, rel_paths, "train")
    X_va, y_va = build_arrays(assign, feat_by_path, nh_by_path, rel_paths, "val")
    X_te, y_te = build_arrays(assign, feat_by_path, nh_by_path, rel_paths, "test")

    print(f"train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    # Augment training data
    X_tr_aug, y_tr_aug = augment(X_tr, y_tr)
    print(f"augmented train: {len(X_tr_aug)} (3x original)")

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42)),
    ])
    clf.fit(X_tr_aug, y_tr_aug)

    for name, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        macro = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
        print(f"{name}: accuracy={acc:.4f}  macro-F1={macro:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")

if __name__ == "__main__":
    main()
