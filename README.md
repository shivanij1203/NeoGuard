# NeoGuard

Real-time pain monitoring for NICU neonates. Uses facial expression analysis and cry audio to detect pain continuously, instead of relying on nurses doing manual NIPS scores every few hours.

## How it works

Camera captures the baby's face, MediaPipe extracts 468 facial landmarks, then we compute geometric features that map to pain-relevant Action Units (brow furrow, eye squeeze, mouth stretch, etc). Those go into an XGBoost classifier that outputs a 0-10 pain score.

On the audio side, microphone input gets run through librosa (MFCCs, spectral features, pitch) into another classifier that tells pain cries apart from hunger/tired cries.

Both scores combine 70/30 (face/audio) into a composite, and the dashboard shows it in real-time over WebSocket. Nurses get alerted when it crosses threshold.

## Running it

```bash
# backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# frontend
cd frontend
npm install
npm run dev

# or just docker
docker-compose up --build
```

Training (if you want to retrain from scratch):
```bash
python ml_training/scripts/download_datasets.py
python ml_training/scripts/train_models.py --model all
```

## Stack

FastAPI, SQLAlchemy, SQLite, MediaPipe, OpenCV, scikit-learn, XGBoost, librosa, React, Vite, Tailwind, Recharts, WebSockets, Docker
