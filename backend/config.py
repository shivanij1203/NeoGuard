import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).parent


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    app_name: str = "NeoGuard"
    debug: bool = os.environ.get("NEOGUARD_DEBUG", "1") == "1"
    database_url: str = os.environ.get(
        "NEOGUARD_DATABASE_URL", "sqlite+aiosqlite:///./neoguard.db"
    )

    # NIPS-aligned alert cutoffs.
    pain_alert_threshold: int = _env_int("NEOGUARD_PAIN_ALERT_THRESHOLD", 4)
    pain_urgent_threshold: int = _env_int("NEOGUARD_PAIN_URGENT_THRESHOLD", 7)

    # Modality weights. Facial signal carried more variance on the held-out test
    # set than cry audio (cry is silent for ~30% of pain events in preemies),
    # so facial is weighted higher. Re-tune via env if you retrain on new data.
    facial_weight: float = _env_float("NEOGUARD_FACIAL_WEIGHT", 0.65)
    audio_weight: float = _env_float("NEOGUARD_AUDIO_WEIGHT", 0.35)

    min_detection_confidence: float = _env_float("NEOGUARD_MIN_DETECTION_CONF", 0.5)
    min_tracking_confidence: float = _env_float("NEOGUARD_MIN_TRACKING_CONF", 0.5)

    models_dir: Path = BACKEND_DIR / "models"
    facial_model_path: Path = BACKEND_DIR / "models" / "facial_pain_clf.joblib"
    cry_model_path: Path = BACKEND_DIR / "models" / "cry_clf.joblib"

    ws_broadcast_interval: float = 0.5
    audio_sample_rate: int = 22050
    audio_duration: float = 3.0


settings = Settings()
