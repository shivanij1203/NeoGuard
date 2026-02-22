# NeoGuard — Neonatal Pain Detection System

AI-powered continuous pain monitoring for NICU neonates using facial expression analysis and cry audio classification.

## The Problem

Premature and critically ill neonates in NICUs experience frequent painful procedures but are often too weak to cry. Current pain assessment relies on sporadic manual scoring by nurses. The only commercial solution (PainChek) does 3-second snapshots. NeoGuard provides **continuous real-time monitoring** with automatic nurse alerts.

## How It Works

```
Camera Feed → MediaPipe Face Mesh → AU-proxy Features → Pain Classifier ─┐
                                                                          ├─→ Composite Score → Dashboard + Alerts
Microphone  → librosa Features    → Cry Classifier    → Pain/Non-pain ──┘
```

### Facial Pain Detection
- **MediaPipe Face Mesh** extracts 468 facial landmarks in real-time
- Geometric features map to neonatal pain Action Units (AU4, AU6+7, AU9+10, AU43, AU27)
- XGBoost/RandomForest classifier produces pain score 0-10

### Cry Audio Classification
- **librosa** extracts MFCCs, spectral centroid, F0, RMS energy
- XGBoost classifier distinguishes pain cries from hunger/tired/discomfort
- Trained on 3 Kaggle infant cry datasets (294 MB total)

### Composite Scoring (NIPS-Inspired)
| Score | Level | Action |
|-------|-------|--------|
| 0-1 | No Pain (green) | — |
| 2-3 | Mild Discomfort (yellow) | Monitor |
| 4-6 | Moderate Pain (orange) | Notify nurse |
| 7-10 | Severe Pain (red) | Urgent alert |

**Weights:** Facial 70% + Audio 30%

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite + TailwindCSS + Recharts |
| Backend | FastAPI + Python 3.11 + SQLAlchemy + SQLite |
| Real-time | WebSockets |
| CV | MediaPipe Face Mesh + OpenCV |
| ML | scikit-learn + XGBoost |
| Audio | librosa |
| Deployment | Docker Compose |

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Kaggle API credentials (for dataset download)

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Train Models
```bash
# Download datasets
python ml_training/scripts/download_datasets.py

# Train classifiers
python ml_training/scripts/train_models.py --model all
```

### Docker
```bash
docker-compose up --build
```

## API Documentation

Once running, visit `http://localhost:8000/docs` for interactive Swagger docs.

### Key Endpoints
- `GET /api/patients/` — List patients
- `POST /api/patients/` — Add patient
- `GET /api/scores/{patient_id}` — Pain score history
- `WS /ws/monitor/{patient_id}` — Real-time monitoring
- `WS /ws/dashboard` — Dashboard broadcast feed

## Project Structure

```
NeoGuard/
├── backend/          # FastAPI + ML pipeline
│   ├── ml/           # Face detector, feature extractor, classifiers, scoring
│   ├── routers/      # REST + WebSocket endpoints
│   └── db/           # SQLAlchemy models
├── frontend/         # React dashboard
│   └── src/components/  # PainGauge, PainChart, CameraFeed, etc.
├── ml_training/      # Training scripts + notebooks
└── data/             # Datasets (not committed)
```

## Impact

- **1 in 10 babies** need NICU care
- Premature infants undergo **10-18 painful procedures daily**
- Many are too weak to cry — their pain goes undetected
- NeoGuard fills this gap with continuous, AI-powered monitoring
