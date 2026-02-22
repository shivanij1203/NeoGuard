#!/usr/bin/env python3
"""
NeoGuard Model Training CLI

Trains both the facial pain classifier and cry audio classifier.

Usage:
    python train_models.py --model facial    # Train facial pain classifier
    python train_models.py --model cry       # Train cry audio classifier
    python train_models.py --model all       # Train both
"""

import argparse
import sys
import os
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Add backend to path for imports
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = BACKEND_DIR / "models"


def train_cry_classifier():
    """
    Train XGBoost classifier on infant cry audio features.
    Binary classification: pain cry (1) vs non-pain cry (0).
    """
    print("\n" + "=" * 60)
    print("Training Cry Audio Classifier")
    print("=" * 60)

    from ml.cry_analyzer import CryAnalyzer
    import librosa

    analyzer = CryAnalyzer()

    # Pain-related categories
    pain_categories = {"belly_pain", "discomfort", "pain", "colic"}
    non_pain_categories = {"hungry", "tired", "burping", "lonely", "scared", "sleepy", "awake", "hug", "cold_hot"}

    raw_dir = DATA_DIR / "raw"
    features_list = []
    labels = []

    # Process each dataset
    for dataset_name in ["infant_cry_corpus", "infant_cry_dataset", "baby_cry_sense"]:
        dataset_dir = raw_dir / dataset_name
        if not dataset_dir.exists():
            print(f"  [SKIP] {dataset_name} not found at {dataset_dir}")
            continue

        print(f"\n  Processing {dataset_name}...")

        # Walk through category folders
        for category_dir in sorted(dataset_dir.rglob("*")):
            if not category_dir.is_dir():
                continue

            category = category_dir.name.lower().replace(" ", "_").replace("-", "_")

            # Determine label
            if category in pain_categories:
                label = 1
            elif category in non_pain_categories:
                label = 0
            else:
                continue

            audio_files = list(category_dir.glob("*.wav")) + \
                          list(category_dir.glob("*.mp3")) + \
                          list(category_dir.glob("*.ogg"))

            print(f"    {category}: {len(audio_files)} files → {'PAIN' if label == 1 else 'NON-PAIN'}")

            for audio_file in audio_files:
                try:
                    audio, sr = librosa.load(str(audio_file), sr=22050, mono=True, duration=5.0)
                    feats = analyzer.extract_features(audio, sr)
                    features_list.append(feats)
                    labels.append(label)
                except Exception as e:
                    print(f"      [ERROR] {audio_file.name}: {e}")

    if len(features_list) == 0:
        print("\n  [ERROR] No audio files processed. Download datasets first:")
        print("    python ml_training/scripts/download_datasets.py")
        return

    X = np.array(features_list)
    y = np.array(labels)

    print(f"\n  Total samples: {len(y)} (pain: {sum(y)}, non-pain: {len(y) - sum(y)})")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    print("\n  Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy: {accuracy:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-pain", "pain"]))

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "cry_clf.joblib"
    scaler_path = MODELS_DIR / "cry_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  Model saved to {model_path}")
    print(f"  Scaler saved to {scaler_path}")


def train_facial_classifier():
    """
    Train facial pain classifier on geometric features.
    Uses synthetic training data from MediaPipe Face Mesh.
    """
    print("\n" + "=" * 60)
    print("Training Facial Pain Classifier")
    print("=" * 60)

    from ml.face_detector import FaceDetector
    from ml.feature_extractor import FeatureExtractor
    import cv2

    detector = FaceDetector()
    extractor = FeatureExtractor()

    # Check for FER2013 or CK+ datasets
    fer_dir = DATA_DIR / "raw" / "fer2013"
    ck_dir = DATA_DIR / "raw" / "ck_plus"

    features_list = []
    labels = []

    # Process facial expression datasets
    # Map expression categories to pain-relevant labels
    # Pain-like: angry, disgust, fear, sad → higher pain proxy
    # Non-pain: happy, surprise, neutral → lower pain proxy
    pain_expressions = {"angry", "disgust", "fear", "sad"}
    neutral_expressions = {"happy", "surprise", "neutral"}

    for dataset_dir in [fer_dir, ck_dir]:
        if not dataset_dir.exists():
            print(f"  [SKIP] {dataset_dir.name} not found")
            continue

        print(f"\n  Processing {dataset_dir.name}...")

        for category_dir in sorted(dataset_dir.rglob("*")):
            if not category_dir.is_dir():
                continue

            category = category_dir.name.lower()
            if category in pain_expressions:
                label_score = 6.0  # Moderate-high pain proxy
            elif category in neutral_expressions:
                label_score = 1.0  # Low/no pain proxy
            else:
                continue

            image_files = list(category_dir.glob("*.png")) + \
                          list(category_dir.glob("*.jpg")) + \
                          list(category_dir.glob("*.jpeg"))

            processed = 0
            for img_file in image_files[:500]:  # Cap per category
                try:
                    frame = cv2.imread(str(img_file))
                    if frame is None:
                        continue

                    # Resize small images for MediaPipe
                    if frame.shape[0] < 100:
                        frame = cv2.resize(frame, (256, 256))

                    detection = detector.detect(frame)
                    if detection is None:
                        continue

                    feats = extractor.extract(detection["landmarks_px"])
                    feat_array = extractor.features_to_array(feats)
                    features_list.append(feat_array)
                    labels.append(label_score)
                    processed += 1
                except Exception as e:
                    pass

            print(f"    {category}: {processed} faces processed → score {label_score}")

    detector.close()

    if len(features_list) == 0:
        print("\n  [WARNING] No dataset images processed.")
        print("  Training with synthetic geometric data instead...")
        _train_facial_synthetic()
        return

    X = np.array(features_list)
    y = np.array(labels)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"\n  MAE: {mae:.3f}")
    print(f"  Score range: {y_pred.min():.2f} - {y_pred.max():.2f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "facial_pain_clf.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")


def _train_facial_synthetic():
    """
    Train on synthetic geometric features when no image dataset is available.
    Generates training data based on known AU-pain relationships.
    """
    print("  Generating synthetic geometric training data...")
    np.random.seed(42)

    feature_names = [
        "brow_eye_dist_norm", "inner_brow_dist_norm", "brow_slope_avg",
        "left_ear", "right_ear", "avg_ear",
        "nose_lip_dist_norm", "nose_length_norm",
        "mouth_aspect_ratio", "mouth_height_norm", "mouth_width_norm", "lip_stretch_norm",
        "eye_asymmetry",
    ]

    n_samples = 2000
    X = []
    y = []

    for _ in range(n_samples):
        pain_level = np.random.uniform(0, 10)

        # Generate features correlated with pain level
        pain_factor = pain_level / 10.0

        features = {
            "brow_eye_dist_norm": 0.08 - 0.04 * pain_factor + np.random.normal(0, 0.005),
            "inner_brow_dist_norm": 0.2 - 0.08 * pain_factor + np.random.normal(0, 0.01),
            "brow_slope_avg": 0.1 + 0.3 * pain_factor + np.random.normal(0, 0.03),
            "left_ear": 0.35 - 0.2 * pain_factor + np.random.normal(0, 0.02),
            "right_ear": 0.35 - 0.2 * pain_factor + np.random.normal(0, 0.02),
            "avg_ear": 0.35 - 0.2 * pain_factor + np.random.normal(0, 0.015),
            "nose_lip_dist_norm": 0.08 - 0.04 * pain_factor + np.random.normal(0, 0.005),
            "nose_length_norm": 0.12 + 0.02 * pain_factor + np.random.normal(0, 0.005),
            "mouth_aspect_ratio": 0.1 + 0.6 * pain_factor + np.random.normal(0, 0.05),
            "mouth_height_norm": 0.05 + 0.1 * pain_factor + np.random.normal(0, 0.01),
            "mouth_width_norm": 0.3 + 0.1 * pain_factor + np.random.normal(0, 0.02),
            "lip_stretch_norm": 0.08 + 0.12 * pain_factor + np.random.normal(0, 0.01),
            "eye_asymmetry": 0.05 + 0.2 * pain_factor + np.random.normal(0, 0.03),
        }

        X.append([features[name] for name in feature_names])
        y.append(pain_level)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="rmse",
        random_state=42,
    )

    # Use regression via XGBRegressor for continuous pain scores
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    print(f"  Synthetic model MAE: {mae:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "facial_pain_clf.joblib"
    joblib.dump(model, model_path)
    print(f"  Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="NeoGuard Model Training")
    parser.add_argument("--model", choices=["facial", "cry", "all"], default="all")
    args = parser.parse_args()

    if args.model in ("facial", "all"):
        train_facial_classifier()

    if args.model in ("cry", "all"):
        train_cry_classifier()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
