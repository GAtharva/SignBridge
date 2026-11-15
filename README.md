# SignBridge — ISL Alphabet Recognizer (Phase 1)

A sign-to-text recognizer for the Indian Sign Language (ISL) fingerspelling
alphabet (A-Z). Phase 1 uses MediaPipe hand landmarks + scikit-learn baselines.
It is **alphabet-only, static (isolated) signs, sign-to-text only**. There is no
text-to-sign, no continuous/sign-sentence recognition, and no Hindi/English word
output.

## Status (honest)

| Component | Status |
|---|---|
| Data | ISL alphabet A-Z, ~26k images (Training/Validation/Testing), read-only |
| Landmark extraction | MediaPipe Tasks `HandLandmarker`, 89.2% hand detection overall |
| Leakage-safe split | Whole hash-groups assigned to train/val/test; zero cross-split sha256/phash |
| Baseline model | RandomForest, trained on new split |
| Test accuracy (no-hand excluded) | **99.13%** (macro-F1 0.9911, n=3,695) |
| Test accuracy (no-hand counted wrong) | **90.13%** (n=4,074; 379 = 9.3% no-hand) |
| Per-class F1 (test, no-hand excluded) | 0.972–1.000 (worst: W=0.972) |
| Backend / frontend / API | None |
| Deployment | None |

**Honest number:** 99.13% excludes images where no hand was detected. If those
are counted as wrong, accuracy drops to 90.13%. The 99.13% figure is likely
slightly inflated by residual same-class near-duplicate landmark poses across
train/test (16.25% of test within epsilon=0.05 of a train sample), even though
exact cross-split duplicates were removed.

This is **not** a like-for-like comparison with the old CNN (previous repo
state), whose 70.72% reported accuracy was inflated by cross-split duplicates
(11.75% exact duplicates); leakage-adjusted old CNN accuracy was 66.60%.

## Setup

Requires Python 3.11 or 3.12 recommended (MediaPipe 1.x works on 3.13 here).

```powershell
# Create the MediaPipe venv (one-time)
py -m venv .venv-mp
.venv-mp\Scripts\python -m pip install --upgrade pip
.venv-mp\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` pins: mediapipe, numpy, pandas, scikit-learn,
opencv-python, Pillow, imagehash, tqdm, joblib, tensorflow, keras.

## Exact run order

```powershell
# 1. Extract MediaPipe landmarks (reads datasets/, writes data/landmarks/)
.venv-mp\Scripts\python signbridge/extract.py

# 2. Leakage-safe re-split (writes data/landmarks/split_assignments.csv)
.venv-mp\Scripts\python signbridge/split.py

# 3. Train baselines, pick best by val macro-F1 (writes models/landmark_baseline.joblib)
.venv-mp\Scripts\python signbridge/train_baseline.py

# 4. Capture your own hold-out set from the webcam (writes data/holdout/)
.venv-mp\Scripts\python scripts/capture_holdout.py   # press A-Z to save, ESC to quit

# 5. Evaluate on your own hold-out set
.venv-mp\Scripts\python scripts/eval_holdout.py

# 6. Live demo (webcam, top-3 probabilities, ESC to quit)
.venv-mp\Scripts\python scripts/live_demo.py
```

`extract.py` streams images and writes in chunks of 1,000; it is resumable via
`data/landmarks/progress.json`. On an 8 GB RAM, no-GPU machine it took
~31 minutes for ~26k images (CPU).

## Folder layout

```
SignBridge/
├── datasets/              (read-only source: raw + processed ISL alphabet images; gitignored)
├── models/
│   ├── hand_landmarker.task   (MediaPipe Tasks model, downloaded)
│   └── landmark_baseline.joblib  (trained best baseline)
├── data/
│   ├── landmarks/
│   │   ├── alphabet.npz          (126-dim feature vectors)
│   │   ├── metadata.csv
│   │   ├── progress.json
│   │   └── split_assignments.csv (leakage-safe train/val/test)
│   └── holdout/
│       └── own_capture.npz       (your webcam captures)
├── reports/phase1/
│   ├── confusion_matrix.png
│   ├── baseline_summary.json
│   └── logs/
├── signbridge/             (package: extract.py, split.py, train_baseline.py)
├── scripts/                (capture_holdout.py, eval_holdout.py, live_demo.py)
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

- Old files (`backend/`, `frontend/`, `notebooks/`, `docs/`, `src/`, `tests/`,
  `models/isl_alphabet_cnn.keras`) are left untouched from the prior prototype
  and are not part of Phase 1.
- `.venv-mp/`, `data/`, `*.npz`, `*.joblib`, `models/*.keras`,
  `reports/phase1/logs/` are gitignored.
- No secrets, API keys, or large tracked files exist in the repo.
