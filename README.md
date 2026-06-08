# NeoGuard

Continuous neonatal pain monitoring research prototype.

**Research prototype, not a medical device.** No claim is made about clinical
readiness. The facial model currently ships with random weights; predictions
are not meaningful until a checkpoint is trained on properly accessed data
under subject-wise cross-validation. Do not use as a clinical assessment.

## Why

NICU neonates undergo frequent painful procedures and are often too weak to
cry visibly. Continuous, multi-modal monitoring is a research direction for
filling the gap between sporadic nurse-scored assessments and silent
distress.

## Pipeline

```
                ┌───────────────────────────────────────────────┐
Camera frame ──▶│ MediaPipe Face Mesh   (presence + occlusion   │
                │                        gate; not a feature    │
                │                        extractor)             │
                └────────────────┬──────────────────────────────┘
                                 │ accepted face crop (RGB, 120x120)
                                 ▼
                ┌───────────────────────────────────────────────┐
                │ N-CNN (three branches: generic 5x5, deep 3x3, │
                │ prominent 7x7; concat; merge conv; classifier)│
                │ Cheap deterministic pass every frame.         │
                │ MC dropout (K passes) on a throttled cadence  │
                │ for uncertainty.                              │
                │ Temperature scaling on the logits.            │
                └────────────────┬──────────────────────────────┘
                                 │ {prob_pain, uncertainty}
                                 ▼
                ┌───────────────────────────────────────────────┐
                │ EMA smoother on prob_pain (holds on absent    │
                │ frames; absent is not zero).                  │
                └────────────────┬──────────────────────────────┘
                                 │
Audio chunk ──▶ Cry analyzer ────┤
                                 ▼
                ┌───────────────────────────────────────────────┐
                │ Uncertainty-weighted fusion                   │
                │   effective_facial_w =                        │
                │     (1 - clip(uncertainty, 0, 1)) * base_w    │
                │   renormalised against audio weight per frame │
                │ Both absent: hold prior composite tagged      │
                │   stale, up to max_stale_age_frames; past     │
                │   the cap, signal_status = unavailable.       │
                └────────────────┬──────────────────────────────┘
                                 │ {composite_score, signal_status,
                                 │  fusion_weights, pain_label}
                                 ▼
                            Dashboard
```

### Facial path (N-CNN)

Clean-room reimplementation of the Zamzmi et al. (IJCNN 2019) N-CNN
topology. Three parallel branches over the same 120x120 RGB face crop:

- **Generic branch** (5x5 conv, broad facial structure)
- **Deep branch** (two stacked 3x3 convs, finer features)
- **Prominent branch** (7x7 conv, salient pain regions: brow, eyes,
  nasolabial fold)

Branches are concatenated along the channel axis (32 + 64 + 32 = 128),
passed through a 3x3 merge conv and pool, flattened, and run through a
two-layer classifier with dropout to two logits. Output is a calibrated
pain probability via temperature scaling, plus an uncertainty estimate
via Monte Carlo dropout. See `backend/ml/ncnn/` and
`NCNN_implementation_spec.md`.

MediaPipe Face Mesh is used **only** as a presence and occlusion gate.
The N-CNN never sees the mesh or any AU-proxy geometric features. The
hand-rolled AU-plus-XGBoost facial scorer has been retired.

### Input-validity gate (out of distribution)

Before any scoring, a detected face is checked against two geometric ratios
derived from the landmarks: forehead-to-chin over temple-to-temple, and
inter-eye-corner over face width. A non-infant face (an adult leaning into
frame, say) trips the gate, and the whole composite hard-stops: no facial
score and no audio score, reported with `signal_status="out_of_distribution"`
and an `ood_reason`. The model has no built-in "not sure" output, so this gate
stands in for one rather than letting the network extrapolate on a subject it
was never meant to score. The two thresholds are unvalidated heuristics (see
`ood_infant_aspect_max` and `ood_infant_ipd_min` in `backend/config.py`) and
need validation against a real infant-versus-adult sample. The gate lives in
`backend/ml/ood_gate.py`, independent of the N-CNN. A known gap remains for the
adult-in-frame-while-infant-cries-offscreen case; it currently fails closed.

### Audio path

Cry analyzer extracts MFCCs, spectral features, and F0; XGBoost classifier
(or a spectral heuristic fallback) emits a `pain` vs `non-pain` cry score.

### Fusion

Uncertainty-weighted: facial weight is scaled by `(1 - clip(uncertainty))`
and the result is renormalised against the audio weight per frame. High
facial uncertainty shifts the composite toward audio; low uncertainty
preserves the facial signal. Both modalities absent hold the prior
composite tagged stale, with a hard cap past which the signal flips to
unavailable rather than presenting a stale number as current.

`alert_level` and `pain_label` are display strings only. They do not page
anyone. A hysteresis-based episode detector for real alerting is deferred
(see `TODO(phase 7)` in `backend/ml/fusion.py` and `backend/ml/scoring.py`)
until trial data can set the thresholds.

## Status

- N-CNN module implemented per spec, with shape and branch-merge tests.
- Calibrated inference path: temperature scaling, MC dropout, `predict_pain`.
- MediaPipe-driven face crop gate with err-toward-None on partial visibility.
- FastAPI wiring: WebSocket carries per-connection cadence and smoother state.
- EMA smoothing and uncertainty-weighted fusion in production code paths.
- Stale composite hold with hard age cap.
- AU-plus-XGBoost facial scorer retired.

What is **not** done:

- Training. There is no checkpoint. The N-CNN currently runs on random
  weights and predictions are not meaningful.
- Real evaluation. No accuracy, F1, AUC, or calibration numbers are
  reported anywhere because no honest number exists yet. Subject-wise
  cross-validation is the only acceptable split when training does happen.
- Episode-detection alerting. Display bands are not alerts.

## Next steps

In order:

1. **Request USF-MNPAD-I access** for real NICU data with NIPS or N-PASS
   labels. This is the gating dependency; nothing meaningful downstream can
   start without it.
2. **Train the N-CNN under the subject-wise CV split** in
   `backend/ml/ncnn/cross_validation.py`, fitting the temperature scaler on a
   subject-disjoint held-out fold. No subject may appear in more than one fold.
   A frame-level split is a bug.
3. **Implement the deferred hysteresis episode detector** for alerting, with
   thresholds set from trial data rather than guessed.

No accuracy, F1, AUC, or calibration metric can be claimed until that path
runs end-to-end against real labelled data under the subject-wise split.

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite + TailwindCSS + Recharts |
| Backend | FastAPI + Python 3.11+ + SQLAlchemy + SQLite |
| Real-time | WebSockets |
| Face gate | MediaPipe Face Mesh + OpenCV |
| Facial classifier | PyTorch (N-CNN, custom) |
| Audio classifier | librosa + XGBoost |
| Tests | pytest |
| Deployment | Docker Compose |

## Quick start

Prerequisites: Python 3.11+, Node 18+.

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The N-CNN initializes with random weights and logs a loud WARNING saying
so. The pipeline runs end-to-end; the numbers it produces are not.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Tests

```bash
cd backend
source venv/bin/activate
pytest tests/
```

### Local webcam pipeline check

```bash
cd backend
source venv/bin/activate
python demo_webcam.py
```

This displays prob_pain, uncertainty, and `frame_to_score_ms` over the
camera feed. It is a pipeline sanity check, nothing more.

### Train the cry classifier

```bash
python ml_training/scripts/train_models.py --model cry
```

Facial training is deliberately not in this script. It belongs in a
subject-wise cross-validation harness that does not exist yet. The previous
synthetic-data facial trainer has been removed.

## API

`http://localhost:8000/docs` once the backend is running.

### Notable endpoints

- `POST /api/analyze/frame`: single-frame analysis. Stateless; MC dropout
  runs on every call.
- `WS /ws/monitor/{patient_id}`: live stream. Owns per-connection
  cadence, smoother, and stale-composite state.
- `WS /ws/dashboard`: broadcast feed.

### Payload fields

- `prob_pain`: raw calibrated probability from the latest frame, or null.
- `prob_pain_smoothed`: EMA-smoothed probability carried across frames.
- `uncertainty`: MC dropout standard deviation. Cached between MC refreshes.
- `facial_score`, `audio_score`, `composite_score`: null when the
  corresponding signal is absent. **Never zero stand-ins.**
- `signal_status`: `fresh`, `facial_only`, `audio_only`, `stale`,
  `unavailable`, or `out_of_distribution`.
- `ood_reason`: set only when `signal_status="out_of_distribution"`, naming
  why the face was rejected (for example `subject_not_infant`).
- `stale`, `stale_age_frames`: set when a held composite is being shown
  because both modalities are currently absent.
- `fusion_weights`: the actual `{facial, audio}` weights used this frame
  after uncertainty scaling and renormalisation.
- `frame_to_score_ms`: end-to-end facial pipeline latency.
- `pain_label`, `alert_level`: **display only**, not clinical and not an
  alert.

## Project layout

```
NeoGuard/
├── backend/
│   ├── ml/
│   │   ├── ncnn/              # N-CNN model, calibration, preprocess,
│   │   │                      # inference, stream state, prob_to_score,
│   │   │                      # subject-wise cross-validation split
│   │   ├── ncnn_classifier.py # facial pipeline wrapper
│   │   ├── face_detector.py   # MediaPipe gate
│   │   ├── ood_gate.py        # infant-versus-adult input-validity gate
│   │   ├── smoother.py        # EMA on prob_pain
│   │   ├── fusion.py          # uncertainty-weighted fusion
│   │   ├── cry_analyzer.py    # audio path
│   │   └── scoring.py         # WebSocket processing
│   ├── routers/
│   └── tests/
├── frontend/
├── ml_training/scripts/       # cry training only
├── standalone/                # standalone NICU image quality filter
└── NCNN_implementation_spec.md
```

## References

- Zamzmi et al. Pain assessment from facial expression: Neonatal
  convolutional neural network (N-CNN). IJCNN 2019.
- Ferreira et al. Revisiting N-CNN for Clinical Practice. PRIME workshop,
  Springer, 2023.
- Ferreira et al. Disclosing neonatal pain in real-time. Computers in
  Biology and Medicine, 2025.
- Salekin et al. USF-MNPAD-I multimodal neonatal pain dataset. Data in
  Brief, 2021.
