"""Phase 1 Step 5: Evaluate the baseline model on your own captured hold-out set.

Loads models/landmark_baseline.joblib and data/holdout/own_capture.npz,
prints accuracy and per-class results.

Run:  .venv-mp/Scripts/python scripts/eval_holdout.py
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

MODEL_PATH = REPO / "models" / "landmark_baseline.joblib"
NPZ_PATH = REPO / "data" / "holdout" / "own_capture.npz"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def main():
    if not MODEL_PATH.exists():
        print(f"FATAL: model not found: {MODEL_PATH} (run train_baseline.py first)")
        sys.exit(1)
    if not NPZ_PATH.exists():
        print(f"FATAL: hold-out capture not found: {NPZ_PATH} (run capture_holdout.py first)")
        sys.exit(1)

    clf = joblib.load(MODEL_PATH)
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        X = z["feat"].tolist()
        y = z["label"].tolist()
    X = np.stack(X)
    y = np.array(y)

    print(f"Loaded {len(y)} hold-out captures")
    print("Counts:", dict(sorted(Counter(y).items())))

    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
    print(f"\nHold-out accuracy: {acc:.4f}  macro-F1: {f1:.4f}")

    print("\nPer-class results:")
    for cls in CLASSES:
        idxs = np.where(y == cls)[0]
        if len(idxs) == 0:
            print(f"  {cls}: no samples")
            continue
        cls_acc = (pred[idxs] == cls).mean()
        print(f"  {cls}: accuracy={cls_acc:.2f}  n={len(idxs)}")

    wrong = [(y[i], pred[i]) for i in range(len(y)) if y[i] != pred[i]]
    if wrong:
        print(f"\nMisclassifications ({len(wrong)}): true->pred")
        for t, p in wrong[:20]:
            print(f"  {t} -> {p}")


if __name__ == "__main__":
    main()
