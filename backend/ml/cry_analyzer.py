import numpy as np
import joblib
import librosa
import logging
from pathlib import Path
from io import BytesIO

from config import settings

logger = logging.getLogger(__name__)


class CryAnalyzer:

    PAIN_LABELS = {"belly_pain", "discomfort", "pain"}
    NON_PAIN_LABELS = {"hungry", "tired", "burping", "lonely", "scared", "sleepy", "awake", "hug"}

    def __init__(self):
        self.model = None
        self.scaler = None
        self.sr = settings.audio_sample_rate
        self._load_model()

    def _load_model(self):
        model_path = settings.cry_model_path
        scaler_path = model_path.parent / "cry_scaler.joblib"
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Loaded cry classifier from {model_path}")
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
        else:
            logger.warning(f"No cry model at {model_path}, using spectral heuristics")

    def extract_features(self, audio: np.ndarray, sr: int | None = None) -> np.ndarray:
        sr = sr or self.sr

        min_samples = int(sr * 0.5)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))

        features = []

        # MFCCs (13 coefficients × mean + std = 26 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))

        # Spectral centroid (mean)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spectral_centroid))

        # Spectral bandwidth (mean)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spectral_bandwidth))

        # Spectral rolloff (mean)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(spectral_rolloff))

        # Zero-crossing rate (mean)
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))

        # RMS energy (mean)
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))

        # Fundamental frequency (F0) estimate
        f0 = librosa.yin(audio, fmin=80, fmax=1000, sr=sr)
        f0_valid = f0[f0 > 0]
        features.append(np.mean(f0_valid) if len(f0_valid) > 0 else 0.0)
        features.append(np.std(f0_valid) if len(f0_valid) > 0 else 0.0)

        return np.array(features, dtype=np.float32)

    def predict(self, audio: np.ndarray, sr: int | None = None) -> dict:
        sr = sr or self.sr

        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            return {
                "cry_detected": False,
                "cry_type": "no_cry",
                "audio_score": 0.0,
                "confidence": 1.0,
            }

        features = self.extract_features(audio, sr)

        if self.model is not None:
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)

            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            is_pain = prediction == 1  # 1 = pain, 0 = non-pain
            confidence = float(np.max(probabilities))

            return {
                "cry_detected": True,
                "cry_type": "pain" if is_pain else "non-pain",
                "audio_score": round(confidence * 10 if is_pain else (1 - confidence) * 3, 2),
                "confidence": round(confidence, 3),
            }
        else:
            return self._heuristic_classify(audio, sr, features)

    def predict_from_bytes(self, audio_bytes: bytes) -> dict:
        audio, sr = librosa.load(BytesIO(audio_bytes), sr=self.sr, mono=True)
        return self.predict(audio, sr)

    def predict_from_file(self, file_path: str | Path) -> dict:
        audio, sr = librosa.load(str(file_path), sr=self.sr, mono=True)
        return self.predict(audio, sr)

    def _heuristic_classify(self, audio: np.ndarray, sr: int, features: np.ndarray) -> dict:
        # spectral heuristics fallback — pain cries have higher pitch + more energy
        mfcc_means = features[:13]
        spectral_centroid = features[26]
        rms_energy = features[30]
        f0_mean = features[31]
        f0_std = features[32]

        pain_score = 0.0

        # High fundamental frequency suggests distress
        if f0_mean > 400:
            pain_score += 3.0
        elif f0_mean > 300:
            pain_score += 1.5

        # High F0 variability suggests pain cry
        if f0_std > 100:
            pain_score += 2.0
        elif f0_std > 50:
            pain_score += 1.0

        # High spectral centroid = more high-frequency energy
        if spectral_centroid > 2000:
            pain_score += 2.0
        elif spectral_centroid > 1500:
            pain_score += 1.0

        # High RMS = louder cry
        if rms_energy > 0.1:
            pain_score += 1.5
        elif rms_energy > 0.05:
            pain_score += 0.5

        pain_score = min(pain_score, 10.0)
        is_pain = pain_score >= 4.0

        return {
            "cry_detected": True,
            "cry_type": "pain" if is_pain else "non-pain",
            "audio_score": round(pain_score, 2),
            "confidence": 0.5,  # Low confidence for heuristic
        }

    def get_feature_names(self) -> list[str]:
        names = []
        for i in range(13):
            names.append(f"mfcc_{i}_mean")
        for i in range(13):
            names.append(f"mfcc_{i}_std")
        names.extend([
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
            "zero_crossing_rate", "rms_energy", "f0_mean", "f0_std"
        ])
        return names
