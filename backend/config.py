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
    cry_model_path: Path = Path(__file__).parent / "models" / "cry_clf.joblib"
    # Facial model checkpoint path. None until a subject-wise CV training run
    # produces one. NCNNFacialPainClassifier currently ships with random
    # weights; predictions are not meaningful until this is set.
    ncnn_checkpoint_path: Path | None = None

    # WebSocket
    ws_broadcast_interval: float = 0.5  # seconds

    # Audio settings
    audio_sample_rate: int = 22050
    audio_duration: float = 3.0  # seconds per analysis window

    # N-CNN settings (research prototype, not a medical device).
    # TODO: confirm against IJCNN 2019 source for any value flagged below.
    ncnn_input_size: int = 120  # TODO: confirm against IJCNN 2019 source
    ncnn_dropout: float = 0.5  # TODO: confirm against IJCNN 2019 source
    ncnn_dense_width: int = 128  # TODO: confirm against IJCNN 2019 source
    # Per-channel normalization placeholders. Zero mean and unit std are a no-op
    # by design. This model trains from scratch, so ImageNet stats are wrong here.
    # TODO: compute mean and std on the training set once data lands.
    ncnn_norm_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ncnn_norm_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # MC dropout cadence: 1 deterministic pass per frame for prob_pain, K-pass
    # MC dropout for uncertainty on a throttle. Both knobs are configurable.
    ncnn_mc_passes: int = 20
    ncnn_mc_interval_frames: int = 30  # roughly once a second at 30 FPS
    # Pain class index in the softmax output. Class 0 = no pain, class 1 = pain.
    ncnn_pain_class_index: int = 1
    # Face crop gate parameters.
    # Margin as a fraction of bbox side, added on each edge before cropping.
    ncnn_crop_margin_ratio: float = 0.15
    # Minimum bbox side (post-margin) in pixels. Smaller crops are rejected.
    ncnn_min_face_size_px: int = 32
    # Exponential moving average smoother on the calibrated prob_pain. Smaller
    # alpha is more smoothing. Operates per frame; at constant FPS, the
    # effective half-life in frames is ln(0.5) / ln(1 - alpha).
    ema_smoother_alpha: float = 0.2
    # Hard cap on how long a both-modalities-absent composite may be held as
    # stale. Past this age the composite flips to None with signal_status
    # unavailable, because a minutes-old number rendered as if current is
    # worse than an honest no-signal. Counted in frames; at ~30 FPS the
    # default below is ~10 seconds.
    max_stale_age_frames: int = 300

    class Config:
        env_prefix = "NEOGUARD_"


settings = Settings()
