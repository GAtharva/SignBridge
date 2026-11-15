"""Phase 1 Step 4: Evaluate v2 baseline on your own captured hold-out set.

Loads models/landmark_v2.joblib and data/holdout/own_capture_v2.npz,
builds features via signbridge/features.py, reports accuracy and per-class
results, plus accuracy on mirrored own captures.

Run: .venv-mp/Scripts/python scripts/eval_v2.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.features import build_features_array, FEATURE_DIM  # noqa: E402

MODEL_PATH = REPO / "models" / "landmark_v2.joblib"
NPZ_PATH = REPO / "data" / "holdout" / "own_capture_v2.npz"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def main():
    if not MODEL_PATH.exists():
        print(f"FATAL: model not found: {MODEL_PATH} (run train_v2.py first)")
        sys.exit(1)
    if not NPZ_PATH.exists():
        print(f"FATAL: hold-out capture not found: {NPZ_PATH} (run capture_v2.py first)")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        raw = z["raw"].tolist()
        present = z["present"].tolist()
        labels = z["label"].tolist()

    raw_arr = np.stack([np.asarray(r, dtype=np.float32) for r in raw])
    present_arr = np.stack([np.asarray(p, dtype=bool) for p in present])
    y = np.array(labels)

    print(f"Loaded {len(y)} hold-out captures")
    print("Counts:", dict(sorted(Counter(y).items())))

    X = build_features_array(raw_arr, present_arr, mirror=False)
    X_mirror = build_features_array(raw_arr, present_arr, mirror=True)

    pred = clf.predict(X)
    pred_mirror = clf.predict(X_mirror)

    acc = accuracy_score(y, pred)
    acc_mirror = accuracy_score(y, pred_mirror)
    macro = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
    macro_mirror = f1_score(y, pred_mirror, average="macro", labels=CLASSES, zero_division=0)

    print(f"\nHold-out accuracy: {acc:.4f}  macro-F1: {macro:.4f}")
    print(f"Hold-out accuracy (mirrored): {acc_mirror:.4f}  macro-F1: {macro_mirror:.4f}")
    print(f"Mirror drop: {acc - acc_mirror:.4f} ({100*(acc-acc_mirror):.2f} points)")

    print("\nPer-class results:")
    for cls in CLASSES:
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0:
            print(f"  {cls}: no samples")
            continue
        cls_acc = (pred[idxs] == cls).mean()
        print(f"  {cls}: accuracy={cls_acc:.2f} n={len(idxs)}")

    wrong = [(y[i], pred[i]) for i in range(len(y)) if y[i] != pred[i]]
    if wrong:
        print(f"\nMisclassifications ({len(wrong)}): true->pred")
        for t, p in wrong[:20]:
            print(f"  {t} -> {p}")


if __name__ == "__main__":
    main()
