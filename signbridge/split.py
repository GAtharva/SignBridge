"""Phase 1 Step 3: Leakage-safe re-split of landmark data.

Groups images that share sha256 OR phash (union-find), then assigns WHOLE groups
to train/val/test (~70/15/15, stratified by class, seed 42). Verifies zero
shared sha256/phash across new splits. Residual landmark-space leak check.

Reads:  data/landmarks/metadata.csv + data/landmarks/alphabet.npz
Writes: data/landmarks/split_assignments.csv
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

REPO = Path(__file__).resolve().parents[1]
LAND_DIR = REPO / "data" / "landmarks"
CSV_PATH = LAND_DIR / "metadata.csv"
NPZ_PATH = LAND_DIR / "alphabet.npz"
OUT_PATH = LAND_DIR / "split_assignments.csv"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42
EPSILON = 0.05  # near-duplicate threshold in normalized landmark space


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_data():
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    with np.load(NPZ_PATH, allow_pickle=True) as z:
        features = z["features"].tolist()
        rel_paths = z["rel_path"].tolist()
    # index by rel_path
    feat_by_path = {p: np.asarray(feat, dtype=np.float32) for p, feat in zip(rel_paths, features)}
    return rows, feat_by_path


def build_groups(rows):
    """Union-find on sha256 OR phash; returns list of groups (list of row indices)."""
    n = len(rows)
    uf = UnionFind(n)
    sha_index = defaultdict(list)
    ph_index = defaultdict(list)
    for i, r in enumerate(rows):
        sha_index[r["sha256"]].append(i)
        ph_index[r["phash"]].append(i)
    for idxs in sha_index.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    for idxs in ph_index.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


def assign_splits(rows, groups):
    """Per-class stratified whole-group split ~70/15/15, seed 42."""
    rng = random.Random(SEED)
    # group label = majority class
    group_label = []
    group_members = []
    for g in groups:
        classes = Counter(rows[i]["cls"] for i in g)
        group_label.append(classes.most_common(1)[0][0])
        group_members.append(g)

    assignment = {}  # row index -> new split
    new_split_sizes = Counter()
    per_class_new = defaultdict(Counter)

    # group classes for stratification
    classes_present = sorted({r["cls"] for r in rows})
    for cls in classes_present:
        cls_groups = [g for g, lbl in zip(groups, group_label) if lbl == cls]
        rng.shuffle(cls_groups)
        n_imgs = sum(len(g) for g in cls_groups)
        train_target = TRAIN_FRAC * n_imgs
        val_target = (TRAIN_FRAC + VAL_FRAC) * n_imgs
        cum = 0
        for gi, g in enumerate(cls_groups):
            cum += len(g)
            if cum <= train_target:
                split = "train"
            elif cum <= val_target:
                split = "val"
            else:
                split = "test"
            for i in g:
                assignment[i] = split
            new_split_sizes[split] += len(g)
            per_class_new[cls][split] += len(g)

    return assignment, new_split_sizes, per_class_new


def verify_no_leakage(rows, assignment):
    """Zero shared sha256 and zero shared phash between any two new splits."""
    by_split = defaultdict(lambda: {"sha256": set(), "phash": set(), "count": 0})
    for i, r in enumerate(rows):
        s = assignment[i]
        by_split[s]["sha256"].add(r["sha256"])
        by_split[s]["phash"].add(r["phash"])
        by_split[s]["count"] += 1

    splits = ["train", "val", "test"]
    ok = True
    for a in splits:
        for b in splits:
            if a >= b:
                continue
            sha_overlap = by_split[a]["sha256"] & by_split[b]["sha256"]
            ph_overlap = by_split[a]["phash"] & by_split[b]["phash"]
            if sha_overlap:
                ok = False
                print(f"LEAK sha256 {a}<->{b}: {len(sha_overlap)} shared")
            if ph_overlap:
                ok = False
                print(f"LEAK phash {a}<->{b}: {len(ph_overlap)} shared")
    return ok, by_split


def residual_leak_check(rows, assignment, feat_by_path):
    """Nearest-train distance for each test sample; report percentiles."""
    train_idxs = [i for i, r in enumerate(rows) if assignment[i] == "train"]
    test_idxs = [i for i, r in enumerate(rows) if assignment[i] == "test"]

    X_train = np.stack([feat_by_path[rows[i]["rel_path"]] for i in train_idxs])
    X_test = np.stack([feat_by_path[rows[i]["rel_path"]] for i in test_idxs])

    print(f"\nResidual leak check: {len(test_idxs)} test vs {len(train_idxs)} train")
    print(f"Epsilon (near-duplicate threshold): {EPSILON}")

    all_min_d = []
    batch = 500
    for start in range(0, len(X_test), batch):
        chunk = X_test[start:start + batch]
        d = euclidean_distances(chunk, X_train)
        all_min_d.append(d.min(axis=1))
    all_min_d = np.concatenate(all_min_d)

    p_min = all_min_d.min()
    p1 = np.percentile(all_min_d, 1)
    p5 = np.percentile(all_min_d, 5)
    p50 = np.percentile(all_min_d, 50)
    n_eps = int((all_min_d < EPSILON).sum())
    print(f"min distance to nearest train: {p_min:.6f}")
    print(f"1st percentile: {p1:.6f}")
    print(f"5th percentile: {p5:.6f}")
    print(f"50th percentile (median): {p50:.6f}")
    print(f"Test samples within epsilon={EPSILON} of a train sample: {n_eps} "
          f"({n_eps/len(test_idxs)*100:.2f}% of test)")
    return all_min_d


def main():
    if not CSV_PATH.exists() or not NPZ_PATH.exists():
        print("FATAL: run extract.py first (metadata.csv / alphabet.npz missing)")
        sys.exit(1)

    rows, feat_by_path = load_data()
    print(f"Loaded {len(rows)} landmark records")

    groups = build_groups(rows)
    print(f"Union-find groups: {len(groups)} (from {len(rows)} images)")

    assignment, new_split_sizes, per_class_new = assign_splits(rows, groups)

    # Write assignments CSV
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "orig_split", "cls", "new_split", "sha256", "phash", "num_hands"])
        for i, r in enumerate(rows):
            w.writerow([r["rel_path"], r["split"], r["cls"], assignment[i],
                        r["sha256"], r["phash"], r["num_hands"]])
    print(f"\nWrote {OUT_PATH}")

    print("\nNew split sizes:", dict(new_split_sizes))
    total = sum(new_split_sizes.values())
    for s in ["train", "val", "test"]:
        print(f"  {s}: {new_split_sizes[s]} ({new_split_sizes[s]/total*100:.1f}%)")

    print("\nPer-class new split sizes (train/val/test) vs original (Training/Validation/Testing):")
    orig_per_class = defaultdict(Counter)
    for r in rows:
        orig_per_class[r["cls"]][r["split"].lower()] += 1
    for cls in sorted(per_class_new):
        nc = per_class_new[cls]
        oc = orig_per_class[cls]
        print(f"  {cls}: new(train={nc['train']}, val={nc['val']}, test={nc['test']}) "
              f"orig(Training={oc['training']}, Validation={oc['validation']}, Testing={oc['testing']})")

    ok, by_split = verify_no_leakage(rows, assignment)
    print("\nCross-split leakage verification:", "PASS" if ok else "FAIL")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {by_split[s]['count']} images, "
              f"{len(by_split[s]['sha256'])} unique sha256, {len(by_split[s]['phash'])} unique phash")

    residual_leak_check(rows, assignment, feat_by_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
