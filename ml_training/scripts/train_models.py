#!/usr/bin/env python3
"""
NeoGuard Model Training CLI

Trains the cry audio classifier. The facial pain classifier is now the
N-CNN (see backend/ml/ncnn/); facial training belongs in a separate
subject-wise cross-validation harness once real data is available, and is
intentionally not implemented here. The old AU plus XGBoost facial trainer
was removed because it trained on synthetic data and reported results, which
this project does not do.

Usage:
    python train_models.py --model cry       # Train cry audio classifier
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


def main():
    parser = argparse.ArgumentParser(description="NeoGuard Model Training")
    parser.add_argument("--model", choices=["cry"], default="cry")
    args = parser.parse_args()

    if args.model == "cry":
        train_cry_classifier()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
