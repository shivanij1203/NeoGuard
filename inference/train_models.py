# Trains XGBRegressor (video pain) and XGBClassifier (cry detection)
# on synthetic data since real NICU data is PHI-restricted.

import numpy as np
import pickle
import os
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_SAMPLES = 2000
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_video_data(n: int):
    # synthetic AU features correlated with pain level (NIPS/NFCS-based)
    X = np.zeros((n, 13), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pain = np.random.uniform(0, 10)
        p = pain / 10.0  # normalized [0,1]

        noise = lambda scale=0.05: np.random.normal(0, scale)

        # Feature layout (matches feature_extractor.py order):
        # 0: AU1  inner brow raise (gap between inner brows)
        #    At rest ~1.2, pain compresses brows → decreases
        X[i, 0]  = 1.2 - 0.4 * p + noise(0.08)

        # 1: AU4 brow lower L (inner brow to eye inner, smaller = more lowered)
        X[i, 1]  = 0.7 - 0.25 * p + noise(0.06)

        # 2: AU4 brow lower R
        X[i, 2]  = 0.7 - 0.25 * p + noise(0.06)

        # 3: AU2 outer brow raise L (larger = more raised, neutral in pain)
        X[i, 3]  = 0.8 + 0.1 * p + noise(0.07)

        # 4: AU2 outer brow raise R
        X[i, 4]  = 0.8 + 0.1 * p + noise(0.07)

        # 5: AU6 cheek raise L (eye-to-mouth distance, smaller = cheek raised)
        X[i, 5]  = 2.0 - 0.3 * p + noise(0.1)

        # 6: AU6 cheek raise R
        X[i, 6]  = 2.0 - 0.3 * p + noise(0.1)

        # 7: AU9 nose wing spread (larger = more wrinkle)
        X[i, 7]  = 0.9 + 0.4 * p + noise(0.07)

        # 8: AU9 nose bridge-to-tip (shorter = scrunched)
        X[i, 8]  = 0.8 - 0.2 * p + noise(0.06)

        # 9: AU20 lip stretch / mouth width (larger = stretched sideways)
        X[i, 9]  = 1.0 + 0.5 * p + noise(0.08)

        # 10: AU25 lips part / mouth open (larger = open for cry)
        X[i, 10] = 0.1 + 0.6 * p + noise(0.07)

        # 11: AU43 eye closure L (smaller = closed/squeezed)
        X[i, 11] = 0.3 - 0.15 * p + noise(0.04)

        # 12: AU43 eye closure R
        X[i, 12] = 0.3 - 0.15 * p + noise(0.04)

        y[i] = np.clip(pain + np.random.normal(0, 0.5), 0, 10)

    return X, y


def train_video_model():
    print("─" * 50)
    print("Training XGBRegressor (video → pain score)...")

    X, y = generate_video_data(N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  MAE on test set: {mae:.3f} (out of 10)")

    path = os.path.join(MODELS_DIR, "pain_video_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")
    return model


def generate_audio_data(n: int):
    # synthetic audio features — cry vs non-cry
    X = np.zeros((n, 33), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)

    for i in range(n):
        is_cry = np.random.random() > 0.5
        c = 1.0 if is_cry else 0.0

        noise = lambda scale=1.0: np.random.normal(0, scale)

        # MFCCs: cry tends to have higher energy (more negative first MFCC
        # is conventional, but we use absolute-style synthetic values)
        mfcc_base = -50 + 80 * c   # increases with cry
        mfcc_means = np.array([
            mfcc_base + k * 3 + noise(5) for k in range(13)
        ], dtype=np.float32)

        mfcc_stds = np.array([
            5 + 15 * c + noise(3) for _ in range(13)
        ], dtype=np.float32)

        spectral_centroid = 1500 + 2000 * c + noise(300)
        spectral_bandwidth = 1000 + 1500 * c + noise(200)
        spectral_rolloff   = 3000 + 4000 * c + noise(500)
        zcr                = 0.05 + 0.25 * c + noise(0.03)
        rms                = 0.02 + 0.15 * c + noise(0.02)
        f0_mean            = 200  + 400  * c + noise(50)
        f0_std             = 20   + 80   * c + noise(15)

        X[i] = np.concatenate([
            mfcc_means,
            mfcc_stds,
            [spectral_centroid, spectral_bandwidth, spectral_rolloff,
             zcr, rms, f0_mean, f0_std]
        ]).astype(np.float32)
        y[i] = int(is_cry)

    return X, y


def train_audio_model():
    print("─" * 50)
    print("Training XGBClassifier (audio → cry detection)...")

    X, y = generate_audio_data(N_SAMPLES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy on test set: {acc:.3f}")
    print(classification_report(y_test, preds, target_names=["no_cry", "cry"]))

    path = os.path.join(MODELS_DIR, "cry_audio_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")
    return model


if __name__ == "__main__":
    print("NeoGuard — Training Pain Detection Models")
    print("=" * 50)
    print("Note: Models trained on synthetic data (real NICU data is PHI-restricted)")
    print()
    train_video_model()
    print()
    train_audio_model()
    print()
    print("Done. Models saved to inference/models/")
