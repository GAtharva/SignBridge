"""Phase 1 Step 4: Baseline classifiers on leakage-safe landmark splits.

Trains LogisticRegression, RandomForest, MLPClassifier on the NEW train split,
tunes/model-selects on val, reports ONCE on test. Saves best (val macro-F1)
to models/landmark_baseline.joblib.

Reads:  data/landmarks/split_assignments.csv + data/landmarks/alphabet.npz
Writes: models/landmark_baseline.joblib, reports/phase1/confusion_matrix.png
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

REPO = Path(__file__).resolve().parents[1]
LAND_DIR = REPO / "data" / "landmarks"
CSV_PATH = LAND_DIR / "split_assignments.csv"
NPZ_PATH = LAND_DIR / "alphabet.npz"
MODEL_PATH = REPO / "models" / "landmark_baseline.joblib"
CM_PATH = REPO / "reports" / "phase1" / "confusion_matrix.png"

SEED = 42
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def load_data():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        features = z["features"].tolist()
        rel_paths = z["rel_path"].tolist()
    feat_by_path = {p: np.asarray(feat, dtype=np.float32) for p, feat in zip(rel_paths, features)}
    return rows, feat_by_path


def split_arrays(rows, feat_by_path, new_split, include_no_hand):
    X, y, paths = [], [], []
    for i, r in enumerate(rows):
        if r["new_split"] != new_split:
            continue
        if r["num_hands"] == "0" and not include_no_hand:
            continue  # exclude no-hand (zero-vector) images
        feat = feat_by_path[r["rel_path"]]
        X.append(feat)
        y.append(r["cls"])
        paths.append(r["rel_path"])
    return np.stack(X) if X else np.zeros((0, 126)), np.array(y), paths


def main():
    if not CSV_PATH.exists() or not NPZ_PATH.exists():
        print("FATAL: run extract.py and split.py first")
        sys.exit(1)

    rows, feat_by_path = load_data()
    print(f"Loaded {len(rows)} records")

    # Train/val: exclude no-hand (zero-vector) images (uninformative, would pollute).
    X_tr, y_tr, _ = split_arrays(rows, feat_by_path, "train", include_no_hand=False)
    X_va, y_va, _ = split_arrays(rows, feat_by_path, "val", include_no_hand=False)
    # Test: keep ALL images (incl. no-hand) for metric (a); exclude for metric (b).
    X_te_all, y_te_all, te_paths = split_arrays(rows, feat_by_path, "test", include_no_hand=True)
    test_rows = [r for r in rows if r["new_split"] == "test"]
    te_no_hand = np.array([r["num_hands"] == "0" for r in test_rows])

    print(f"train: {X_tr.shape[0]}, val: {X_va.shape[0]}, test(all): {X_te_all.shape[0]}, "
          f"test(no-hand excluded): {(~te_no_hand).sum()}, test(no-hand only): {te_no_hand.sum()}")

    models = {
        "LogisticRegression": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=1)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=2),
        "MLPClassifier": Pipeline([
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                  random_state=SEED, early_stopping=True)),
        ]),
    }

    results = {}
    for name, clf in models.items():
        print(f"\nTraining {name} ...")
        clf.fit(X_tr, y_tr)
        val_pred = clf.predict(X_va)
        val_f1 = f1_score(y_va, val_pred, average="macro", labels=CLASSES, zero_division=0)
        val_acc = accuracy_score(y_va, val_pred)
        print(f"  val accuracy={val_acc:.4f}  val macro-F1={val_f1:.4f}")
        results[name] = (clf, val_f1, val_acc)

    # Select best by val macro-F1
    best_name = max(results, key=lambda k: results[k][1])
    best_clf, best_val_f1, best_val_acc = results[best_name]
    print(f"\nBEST (by val macro-F1): {best_name}  val_macroF1={best_val_f1:.4f}  val_acc={best_val_acc:.4f}")

    # Test evaluation — metric (a): no-hand counted as wrong
    te_pred_all = best_clf.predict(X_te_all)
    te_acc_all = accuracy_score(y_te_all, te_pred_all)
    te_f1_all = f1_score(y_te_all, te_pred_all, average="macro", labels=CLASSES, zero_division=0)

    # Metric (b): exclude no-hand
    X_te = X_te_all[~te_no_hand]
    y_te = y_te_all[~te_no_hand]
    te_pred = best_clf.predict(X_te)
    te_acc = accuracy_score(y_te, te_pred)
    te_f1 = f1_score(y_te, te_pred, average="macro", labels=CLASSES, zero_division=0)

    print(f"\n=== TEST (honest number = no-hand EXCLUDED) ===")
    print(f"Accuracy (no-hand excluded): {te_acc:.4f}  macro-F1: {te_f1:.4f}  n={len(y_te)}")
    print(f"Accuracy (no-hand counted wrong): {te_acc_all:.4f}  macro-F1: {te_f1_all:.4f}  n={len(y_te_all)}")
    print(f"No-hand test images: {te_no_hand.sum()} / {len(y_te_all)} "
          f"({te_no_hand.sum()/len(y_te_all)*100:.1f}%)")

    # Per-class F1 on the honest test set
    report = classification_report(y_te, te_pred, labels=CLASSES, output_dict=True,
                                   zero_division=0)
    print("\nPer-class F1 (test, no-hand excluded):")
    f1s = []
    for cls in CLASSES:
        f1s.append((cls, report[cls]["f1-score"], report[cls]["support"]))
    for cls, f1, sup in f1s:
        print(f"  {cls}: F1={f1:.3f}  support={sup}")

    # Confusion matrix
    cm = confusion_matrix(y_te, te_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
           xticklabels=CLASSES, yticklabels=CLASSES,
           ylabel="True", xlabel="Predicted",
           title=f"ISL Alphabet — {best_name} (test, no-hand excluded)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    REPO.joinpath("reports/phase1").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved: {CM_PATH}")

    # 5 most confused pairs (off-diagonal)
    conf = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                conf.append((cm[i, j], CLASSES[i], CLASSES[j]))
    conf.sort(reverse=True)
    print("\n5 most confused (true->pred, count):")
    for cnt, t, p in conf[:5]:
        print(f"  {t} -> {p}: {cnt}")

    # Save best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODEL_PATH)
    print(f"\nBest model saved: {MODEL_PATH}")

    # Summary JSON
    summary = {
        "best_model": best_name,
        "val_macro_f1": round(best_val_f1, 4),
        "val_accuracy": round(best_val_acc, 4),
        "test_accuracy_no_hand_excluded": round(te_acc, 4),
        "test_macro_f1_no_hand_excluded": round(te_f1, 4),
        "test_accuracy_no_hand_wrong": round(te_acc_all, 4),
        "test_n_excluded": int(len(y_te)),
        "test_n_all": int(len(y_te_all)),
        "no_hand_test": int(te_no_hand.sum()),
    }
    with open(REPO / "reports/phase1" / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved: reports/phase1/baseline_summary.json")


if __name__ == "__main__":
    main()
