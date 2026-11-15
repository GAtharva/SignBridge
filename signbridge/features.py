"""Phase 1 Step 2: Single source of truth for landmark features.

build_features(raw, present, mirror=False) -> float32 vector of length 130.

Per hand (21 landmarks x,y,z):
  - wrist-origin: subtract landmark 0 (wrist) from all landmarks
  - scale: divide by wrist-to-middle-finger-MCP (landmark 9) distance
  - flattened to 63 values
Slot 0 then slot 1 (zeros if absent).

Extra features:
  - n_hands: 0/1/2 (1 value)
  - for 2 hands: (wrist1 - wrist0) / scale_of_slot0_hand (3 values), else zeros

Total = 63 + 63 + 1 + 3 = 130.

Mirror: x -> 1-x on raw FIRST, then re-sort hands by wrist x ascending, then continue
with the normal pipeline. This makes mirroring a no-op on the final features
(up to float precision) because slot order is determined by image position, not
MediaPipe handedness labels.

raw:   float32 array shape (2, 21, 3) - image-normalized landmarks exactly as
       MediaPipe returns them.
present: bool array shape (2,) - True if a hand was detected in that slot.
"""
from __future__ import annotations

import numpy as np

FEATURE_DIM = 130
HAND_DIM = 63  # 21 landmarks * 3 (x, y, z)
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def _hand_feature(pts):
    """Convert raw (21,3) landmarks to wrist-origin scaled 63-vector.

    Returns zeros if all points are zero (hand absent).
    """
    if pts is None or not np.any(pts):
        return np.zeros(HAND_DIM, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float64)
    wrist = pts[WRIST_IDX]
    mid_mcp = pts[MIDDLE_MCP_IDX]
    scale = float(np.linalg.norm(mid_mcp - wrist))
    if scale == 0.0 or not np.isfinite(scale):
        scale = 1.0
    norm = (pts - wrist) / scale
    return norm.reshape(-1).astype(np.float32)


def build_features(raw, present, mirror=False):
    """Build a 130-dim slot-invariant feature vector.

    Args:
        raw: (2, 21, 3) float32 raw landmarks (image-normalized).
        present: (2,) bool - which slots have a hand.
        mirror: if True, x -> 1-x on raw first, then re-sort by wrist x.

    Returns:
        float32 vector length 130.
    """
    raw = np.asarray(raw, dtype=np.float32)
    present = np.asarray(present, dtype=bool)

    if mirror:
        raw = _apply_mirror(raw, present)

    # Determine hand order by wrist x ascending (slot 0 = leftmost in image)
    order = _wrist_x_order(raw, present)

    feat = np.zeros(FEATURE_DIM, dtype=np.float32)
    n_hands = int(present.sum())

    # Per-hand features
    for slot, src_slot in enumerate(order):
        if present[src_slot]:
            feat[slot * HAND_DIM:(slot + 1) * HAND_DIM] = _hand_feature(raw[src_slot])

    # Extra features
    feat[126] = np.float32(n_hands)

    if n_hands == 2:
        s0 = raw[order[0]].astype(np.float64)
        s1 = raw[order[1]].astype(np.float64)
        wrist0 = s0[WRIST_IDX]
        wrist1 = s1[WRIST_IDX]
        # scale of slot-0 hand
        mid_mcp0 = s0[MIDDLE_MCP_IDX]
        scale0 = float(np.linalg.norm(mid_mcp0 - wrist0))
        if scale0 == 0.0 or not np.isfinite(scale0):
            scale0 = 1.0
        diff = (wrist1 - wrist0) / scale0
        feat[127:130] = diff.astype(np.float32)

    return feat


def _apply_mirror(raw, present):
    """x -> 1-x on raw, then return re-ordered copy (wrist x ascending)."""
    raw = raw.copy()
    for slot in range(raw.shape[0]):
        if present[slot]:
            raw[slot, :, 0] = 1.0 - raw[slot, :, 0]
    return raw


def _wrist_x_order(raw, present):
    """Return slot indices ordered by wrist x ascending."""
    present_slots = [i for i in range(raw.shape[0]) if present[i]]
    if len(present_slots) <= 1:
        return present_slots
    wrist_x = {}
    for s in present_slots:
        wrist_x[s] = float(raw[s, WRIST_IDX, 0])
    return sorted(present_slots, key=lambda s: wrist_x[s])


def build_features_array(raw_arr, present_arr, mirror=False):
    """Vectorized version. raw_arr: (N,2,21,3), present_arr: (N,2)."""
    n = raw_arr.shape[0]
    out = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    for i in range(n):
        out[i] = build_features(raw_arr[i], present_arr[i], mirror=mirror)
    return out
