"""Phase 1 Step 2: Unit checks for signbridge/features.py.

(i) One hand always lands in slot 0.
(ii) Mirror then mirror returns original features (within 1e-5).
(iii) Swapping input hand order gives identical output.

Run: .venv-mp/Scripts/python scripts/test_features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.features import build_features, _apply_mirror, FEATURE_DIM

CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def make_random_hand(rng):
    """Generate a plausible hand: 21 landmarks in [0,1]^3."""
    return rng.uniform(0, 1, size=(21, 3)).astype(np.float32)


def check_one_hand_slot0():
    """(i) One hand always lands in slot 0."""
    rng = np.random.default_rng(42)
    fails = 0
    for _ in range(200):
        hand = make_random_hand(rng)
        # Put hand in slot 1 only
        raw = np.zeros((2, 21, 3), dtype=np.float32)
        present = np.array([False, True], dtype=bool)
        raw[1] = hand
        f = build_features(raw, present)
        slot0 = f[:63]
        slot1 = f[63:126]
        if np.abs(slot1).sum() > 1e-6:
            fails += 1
        if np.abs(slot0).sum() <= 1e-6:
            fails += 1
    print(f"Check (i) one hand -> slot 0: {'PASS' if fails == 0 else f'FAIL ({fails} fails)'}")
    return fails == 0


def check_double_mirror():
    """(ii) Mirror then mirror returns original features (within 1e-5).

    build_features(raw, mirror=True) mirrors x then computes features.
    Applying mirror=True to the already-mirrored raw should return the
    original features (mirror is an involution).
    """
    rng = np.random.default_rng(7)
    max_diff = 0.0
    fails = 0
    for _ in range(200):
        raw = np.zeros((2, 21, 3), dtype=np.float32)
        present = np.array([True, True], dtype=bool)
        raw[0] = make_random_hand(rng)
        raw[1] = make_random_hand(rng)
        # Ensure distinct wrist x so ordering matters
        raw[0, 0, 0] = 0.3
        raw[1, 0, 0] = 0.7

        f_orig = build_features(raw, present, mirror=False)
        raw_mirrored = _apply_mirror(raw.copy(), present)
        f_double = build_features(raw_mirrored, present, mirror=True)  # mirror back to original
        diff = np.abs(f_orig - f_double).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            fails += 1
    print(f"Check (ii) double mirror == original: {'PASS' if fails == 0 else f'FAIL ({fails} fails)'} "
          f"(max_diff={max_diff:.2e})")
    return fails == 0


def check_swap_invariance():
    """(iii) Swapping input hand order gives identical output."""
    rng = np.random.default_rng(99)
    fails = 0
    max_diff = 0.0
    for _ in range(200):
        h0 = make_random_hand(rng)
        h1 = make_random_hand(rng)
        # Ensure different wrist x so ordering matters
        h0[0, 0] = 0.2
        h1[0, 0] = 0.8
        raw_a = np.zeros((2, 21, 3), dtype=np.float32)
        present = np.array([True, True], dtype=bool)
        raw_a[0] = h0
        raw_a[1] = h1
        raw_b = raw_a.copy()
        raw_b[0] = h1
        raw_b[1] = h0
        f_a = build_features(raw_a, present, mirror=False)
        f_b = build_features(raw_b, present, mirror=False)
        diff = np.abs(f_a - f_b).max()
        max_diff = max(max_diff, diff)
        if diff > 1e-5:
            fails += 1
    print(f"Check (iii) swap invariance: {'PASS' if fails == 0 else f'FAIL ({fails} fails)'} "
          f"(max_diff={max_diff:.2e})")
    return fails == 0


def check_feature_dim():
    """Feature dimension is exactly 130."""
    rng = np.random.default_rng(1)
    raw = np.zeros((2, 21, 3), dtype=np.float32)
    present = np.array([True, False], dtype=bool)
    raw[0] = make_random_hand(rng)
    f = build_features(raw, present)
    ok = f.shape == (FEATURE_DIM,) and f.dtype == np.float32
    print(f"Check feature dim == 130 and float32: {'PASS' if ok else 'FAIL'} "
          f"(shape={f.shape}, dtype={f.dtype})")
    return ok


def main():
    results = []
    results.append(check_feature_dim())
    results.append(check_one_hand_slot0())
    results.append(check_double_mirror())
    results.append(check_swap_invariance())
    print(f"\n{'ALL PASS' if all(results) else 'SOME FAILED'} ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
