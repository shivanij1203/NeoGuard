from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "NeoGuard"
    debug: bool = True
    database_url: str = "sqlite+aiosqlite:///./neoguard.db"

    # Pain scoring thresholds
    pain_alert_threshold: int = 4
    pain_urgent_threshold: int = 7

    # Composite scoring weights
    facial_weight: float = 0.7
    audio_weight: float = 0.3

    # MediaPipe settings
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Model paths
    models_dir: Path = Path(__file__).parent / "models"
    facial_model_path: Path = Path(__file__).parent / "models" / "facial_pain_clf.joblib"
    cry_model_path: Path = Path(__file__).parent / "models" / "cry_clf.joblib"

    # WebSocket
    ws_broadcast_interval: float = 0.5  # seconds

    # Audio settings
    audio_sample_rate: int = 22050
    audio_duration: float = 3.0  # seconds per analysis window

    class Config:
        env_prefix = "NEOGUARD_"


settings = Settings()
