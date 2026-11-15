"""Phase 1 Step 3: Train slot-invariant RandomForest on v2 raw landmarks.

Uses the SAME new_split assignments from data/landmarks/split_assignments.csv.
Trains RandomForest(n_estimators=200, n_jobs=2, seed 42).
Training set = original + raw-level mirrored copies, no-hand excluded.
Reports: val/test accuracy and macro-F1 (no-hand excluded and counted-wrong),
mirrored-test invariance check computed from RAW-level mirror (real check).
Source hold-out: for each held-out source group, train on ALL images (every
split) from other sources with mirror augmentation, test on ALL images from the
held-out group. Reports classes missing from training and accuracy.
Saves model to models/landmark_v2.joblib, summary to reports/phase1/v2_summary.json.

Run: .venv-mp/Scripts/python signbridge/train_v2.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path, PurePath

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]
NPZ_V2_PATH = REPO / "data" / "landmarks" / "alphabet_v2.npz"
CSV_PATH = REPO / "data" / "landmarks" / "split_assignments.csv"
MODEL_PATH = REPO / "models" / "landmark_v2.joblib"
SUMMARY_PATH = REPO / "reports" / "phase1" / "v2_summary.json"
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
SEED = 42

sys.path.insert(0, str(REPO))
from signbridge.features import build_features_array, FEATURE_DIM  # noqa: E402


def load_v2():
    if not NPZ_V2_PATH.exists():
        print(f"FATAL: {NPZ_V2_PATH} not found. Run extract_v2.py first.")
        sys.exit(1)
    with np.load(NPZ_V2_PATH, allow_pickle=True) as z:
        raw = z["raw"].tolist()
        present = z["present"].tolist()
        rel_path = z["rel_path"].tolist()
        split_arr = z["split"].tolist()
        cls_arr = z["cls"].tolist()
    return raw, present, rel_path, split_arr, cls_arr


def load_assignments():
    assign = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            assign[r["rel_path"]] = r["new_split"]
    return assign


def source_of(stem):
    s = re.sub(r'\(\d+\)$', '', stem)
    if s.isdigit():
        return "numeric"
    m = re.match(r'^([A-Za-z]+)_([A-Za-z]+)_(\d+)$', s)
    if m:
        return m.group(1) + "_" + m.group(2)
    m = re.match(r'^([A-Za-z]+)(\d+)$', s)
    if m:
        return m.group(1)
    return "other"


def build_split_xy(raw_list, present_list, rel_path_list, split_arr, cls_arr,
                   assign, new_split, mirror=False):
    """Build X, y arrays for a given new_split (num_hands>0 only)."""
    Xs, ys = [], []
    for i, rp in enumerate(rel_path_list):
        if assign.get(rp) != new_split:
            continue
        if sum(present_list[i]) == 0:
            continue
        Xs.append(build_features_array(
            np.stack([raw_list[i]]), np.stack([present_list[i]]),
            mirror=mirror)[0])
        ys.append(cls_arr[i])
    if not Xs:
        return np.zeros((0, FEATURE_DIM)), np.array([])
    return np.stack(Xs), np.array(ys)


def build_source_xy(raw_list, present_list, rel_path_list, cls_arr, source_cache,
                    include_sources, mirror=False):
    """Build X, y for rows whose source is in include_sources (all splits)."""
    Xs, ys = [], []
    for i, rp in enumerate(rel_path_list):
        if source_cache.get(rp) not in include_sources:
            continue
        if sum(present_list[i]) == 0:
            continue
        Xs.append(build_features_array(
            np.stack([raw_list[i]]), np.stack([present_list[i]]),
            mirror=mirror)[0])
        ys.append(cls_arr[i])
    if not Xs:
        return np.zeros((0, FEATURE_DIM)), np.array([])
    return np.stack(Xs), np.array(ys)


def evaluate(clf, X, y, label):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    macro = f1_score(y, pred, average="macro", labels=CLASSES, zero_division=0)
    print(f"  {label}: n={len(y)} accuracy={acc:.4f} macro-F1={macro:.4f}")
    return acc, macro, pred


def augment_with_raw_mirror(raw_list, present_list, y_list):
    """Original + raw-level mirrored copies (2x total)."""
    X_orig = build_features_array(
        np.stack(raw_list), np.stack(present_list), mirror=False)
    X_mirr = build_features_array(
        np.stack(raw_list), np.stack(present_list), mirror=True)
    return np.vstack([X_orig, X_mirr]), np.concatenate([y_list, y_list])


def main():
    t0 = time.time()
    print("Loading v2 data ...")
    raw, present, rel_path, split_arr, cls_arr = load_v2()
    assign = load_assignments()
    print(f"Loaded {len(rel_path)} records")

    # Build source for each row
    source_cache = {}
    for i, rp in enumerate(rel_path):
        stem = PurePath(rp).stem
        source_cache[rp] = source_of(stem)

    # ---- Main model: new_split train/val/test ----
    X_tr, y_tr = build_split_xy(raw, present, rel_path, split_arr, cls_arr,
                                assign, "train", mirror=False)
    X_va, y_va = build_split_xy(raw, present, rel_path, split_arr, cls_arr,
                                assign, "val", mirror=False)
    X_te, y_te = build_split_xy(raw, present, rel_path, split_arr, cls_arr,
                                assign, "test", mirror=False)
    print(f"train={len(X_tr)} val={len(X_va)} test={len(X_te)}")

    # Augment training with raw-level mirror (2x)
    X_tr_aug, y_tr_aug = augment_with_raw_mirror(raw, present, y_tr)
    print("augmented train: 2x (original + raw-level mirror)")

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, n_jobs=2,
                                       random_state=SEED)),
    ])
    print("Training ...")
    clf.fit(X_tr_aug, y_tr_aug)

    # Val
    print("\nVAL:")
    va_acc, va_macro, _ = evaluate(clf, X_va, y_va, "val (no-hand excluded)")
    # Val no-hand counted wrong
    va_rows = [(i, rp) for i, rp in enumerate(rel_path)
               if assign.get(rp) == "val"]
    X_va_all, y_va_all = [], []
    for i, rp in va_rows:
        if sum(present[i]) == 0:
            X_va_all.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            y_va_all.append(cls_arr[i])
        else:
            X_va_all.append(build_features_array(
                np.stack([raw[i]]), np.stack([present[i]]))[0])
            y_va_all.append(cls_arr[i])
    X_va_all = np.stack(X_va_all)
    y_va_all = np.array(y_va_all)
    va_acc_all, va_macro_all, _ = evaluate(
        clf, X_va_all, y_va_all, "val (no-hand counted wrong)")

    # Test
    print("\nTEST:")
    te_acc, te_macro, te_pred = evaluate(
        clf, X_te, y_te, "test (no-hand excluded)")
    te_rows = [(i, rp) for i, rp in enumerate(rel_path)
               if assign.get(rp) == "test"]
    X_te_all, y_te_all = [], []
    for i, rp in te_rows:
        if sum(present[i]) == 0:
            X_te_all.append(np.zeros(FEATURE_DIM, dtype=np.float32))
            y_te_all.append(cls_arr[i])
        else:
            X_te_all.append(build_features_array(
                np.stack([raw[i]]), np.stack([present[i]]))[0])
            y_te_all.append(cls_arr[i])
    X_te_all = np.stack(X_te_all)
    y_te_all = np.array(y_te_all)
    te_acc_all, te_macro_all, te_pred_all = evaluate(
        clf, X_te_all, y_te_all, "test (no-hand counted wrong)")

    # Mirrored test invariance - REAL check from raw-level mirror
    X_te_mirror = build_features_array(
        np.stack([raw[i] for i, _ in te_rows]),
        np.stack([present[i] for i, _ in te_rows]), mirror=True)
    te_mirror_acc, te_mirror_macro, _ = evaluate(
        clf, X_te_mirror, y_te, "test MIRRORED (raw-level mirror, no-hand excluded)")

    # Per-class F1 worst 8
    rep = classification_report(y_te, te_pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    per_class = [(c, rep[c]["f1-score"], rep[c]["support"]) for c in CLASSES]
    per_class.sort(key=lambda x: x[1])
    print("\nWorst 8 per-class F1 (test, no-hand excluded):")
    for c, f1, sup in per_class[:8]:
        print(f"  {c}: F1={f1:.3f} support={int(sup)}")

    # Top 5 confusion pairs
    cm = confusion_matrix(y_te, te_pred, labels=CLASSES)
    conf = []
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i, j] > 0:
                conf.append((int(cm[i, j]), CLASSES[i], CLASSES[j]))
    conf.sort(reverse=True)
    print("\nTop 5 confusion pairs (true->pred, count):")
    for cnt, t, p in conf[:5]:
        print(f"  {t} -> {p}: {cnt}")

    # ---- Real source hold-out ----
    print("\n" + "=" * 60)
    print("REAL SOURCE HOLD-OUT (train on all other sources, all splits)")
    print("=" * 60)

    # Define held-out groups
    all_sources = sorted(set(source_cache.values()))
    single_session_sources = [s for s in all_sources
                              if s not in ("numeric", "ary", "anki", "other")]

    holdout_groups = {
        "ary": ["ary"],
        "anki": ["anki"],
        "single_session": single_session_sources,
    }

    holdout_results = []

    for group_name, sources in holdout_groups.items():
        # Train: all images from sources NOT in this group (all splits)
        X_tr_ho, y_tr_ho = build_source_xy(
            raw, present, rel_path, cls_arr, source_cache,
            [s for s in all_sources if s not in sources], mirror=False)
        # Augment with mirror
        raw_list_ho = [raw[i] for i, rp in enumerate(rel_path)
                       if source_cache.get(rp)
                       not in sources and sum(present[i]) > 0]
        present_list_ho = [present[i] for i, rp in enumerate(rel_path)
                           if source_cache.get(rp)
                           not in sources and sum(present[i]) > 0]
        y_list_ho = [cls_arr[i] for i, rp in enumerate(rel_path)
                     if source_cache.get(rp)
                     not in sources and sum(present[i]) > 0]
        X_tr_ho_aug, y_tr_ho_aug = augment_with_raw_mirror(
            raw_list_ho, present_list_ho, y_list_ho)

        # Test: all images from held-out sources (all splits)
        X_te_ho, y_te_ho = build_source_xy(
            raw, present, rel_path, cls_arr, source_cache, sources, mirror=False)

        if len(X_te_ho) == 0:
            print(f"\n  {group_name}: no test images, skipping")
            continue

        clf_ho = Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=2,
                                           random_state=SEED)),
        ])
        clf_ho.fit(X_tr_ho_aug, y_tr_ho_aug)
        ho_acc, ho_macro, ho_pred = evaluate(
            clf_ho, X_te_ho, y_te_ho,
            f"{group_name} hold-out (trained on all other sources)")

        # Classes in test missing from training
        classes_in_train = set(y_tr_ho_aug.tolist())
        classes_in_test = set(y_te_ho.tolist())
        classes_missing = sorted(classes_in_test - classes_in_train)
        print(f"    n_train={len(X_tr_ho_aug)} n_test={len(X_te_ho)} "
              f"classes_missing_from_train={classes_missing}")

        # Per-class accuracy for this hold-out
        rep_ho = classification_report(
            y_te_ho, ho_pred, labels=CLASSES, output_dict=True,
            zero_division=0)
        below_80 = []
        for c in sorted(classes_in_test):
            f1 = rep_ho[c]["f1-score"]
            sup = rep_ho[c]["support"]
            if sup > 0 and f1 < 0.80:
                below_80.append((c, f1, int(sup)))
        if below_80:
            print(f"    Classes with F1 < 0.80: {below_80}")

        holdout_results.append({
            "group": group_name,
            "sources": sources,
            "n_train": len(X_tr_ho_aug),
            "n_test": len(X_te_ho),
            "accuracy": float(ho_acc),
            "macro_f1": float(ho_macro),
            "classes_missing_from_train": classes_missing,
            "classes_below_80_f1": below_80,
        })

    # Numeric-only -> non-numeric (all splits)
    print("\nSOURCE HOLD-OUT (train on numeric only, test on all non-numeric):")
    X_num_train, y_num_train = [], []
    X_non_num_test, y_non_num_test = [], []
    for i, rp in enumerate(rel_path):
        if sum(present[i]) == 0:
            continue
        if source_cache[rp] == "numeric":
            X_num_train.append(raw[i])
            y_num_train.append(cls_arr[i])
        elif source_cache[rp] != "numeric":
            X_non_num_test.append(raw[i])
            y_non_num_test.append(cls_arr[i])
    if X_num_train and X_non_num_test:
        X_num_train_aug, y_num_train_aug = augment_with_raw_mirror(
            X_num_train, [present[i] for i, rp in enumerate(rel_path)
                          if source_cache[rp] == "numeric" and sum(present[i]) > 0],
            y_num_train)
        # Build non-numeric test features
        X_non_te = []
        y_non_te = []
        for i, rp in enumerate(rel_path):
            if source_cache[rp] != "numeric" and sum(present[i]) > 0:
                X_non_te.append(build_features_array(
                    np.stack([raw[i]]), np.stack([present[i]]))[0])
                y_non_te.append(cls_arr[i])
        X_non_te = np.stack(X_non_te)
        y_non_te = np.array(y_non_te)
        common = set(y_num_train_aug) & set(y_non_te)
        mask = np.array([c in common for c in y_non_te])
        X_ct = X_non_te[mask]
        y_ct = y_non_te[mask]
        clf_num = Pipeline([
            ("sc", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=2,
                                           random_state=SEED)),
        ])
        clf_num.fit(X_num_train_aug, y_num_train_aug)
        nn_acc, nn_macro, _ = evaluate(
            clf_num, X_ct, y_ct,
            "non-numeric test (trained on numeric only, all splits)")
        print(f"  Common classes: {sorted(common)}")
    else:
        nn_acc = nn_macro = None
        print("  Insufficient data")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model: {MODEL_PATH}")

    # Summary
    summary = {
        "val_accuracy": float(va_acc),
        "val_macro_f1": float(va_macro),
        "val_accuracy_no_hand_wrong": float(va_acc_all),
        "val_macro_f1_no_hand_wrong": float(va_macro_all),
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_macro),
        "test_accuracy_no_hand_wrong": float(te_acc_all),
        "test_macro_f1_no_hand_wrong": float(te_macro_all),
        "test_mirrored_accuracy": float(te_mirror_acc),
        "test_mirrored_macro_f1": float(te_mirror_macro),
        "holdout_results": holdout_results,
        "numeric_to_non_numeric_accuracy": float(nn_acc) if nn_acc is not None else None,
        "runtime_sec": float(time.time() - t0),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
