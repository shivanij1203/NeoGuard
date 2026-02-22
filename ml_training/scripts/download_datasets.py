#!/usr/bin/env python3
"""
Download cry audio datasets from Kaggle for NeoGuard training.
Requires: pip install kaggle
Setup: Place kaggle.json in ~/.kaggle/ with your API credentials.
"""

import os
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

DATASETS = {
    "infant_cry_corpus": {
        "kaggle_id": "warcoder/infant-cry-audio-corpus",
        "description": "Infant Cry Audio Corpus (41 MB, 9 categories)",
    },
    "infant_cry_dataset": {
        "kaggle_id": "sanmithasadhish/infant-cry-dataset",
        "description": "Infant Cry Dataset (177 MB, 8 categories incl. non-cry)",
    },
    "baby_cry_sense": {
        "kaggle_id": "mennaahmed23/baby-cry-dataset",
        "description": "Baby Cry Sense Dataset (76 MB, 8 categories)",
    },
    "fer2013": {
        "kaggle_id": "msambare/fer2013",
        "description": "FER2013 Facial Expression (35K images, 48x48)",
    },
    "ck_plus": {
        "kaggle_id": "davilsena/ckdataset",
        "description": "CK+ Facial Expression Dataset (920 images)",
    },
}


def download_dataset(name: str, dataset_info: dict):
    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()):
        print(f"  [SKIP] {name} — already downloaded at {dest}")
        return

    print(f"  [DOWNLOADING] {dataset_info['description']}")
    print(f"    → {dest}")

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_info["kaggle_id"],
                "-p", str(dest),
                "--unzip",
            ],
            check=True,
        )
        print(f"  [OK] {name} downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to download {name}: {e}")
    except FileNotFoundError:
        print("  [ERROR] kaggle CLI not found. Install with: pip install kaggle")
        sys.exit(1)


def main():
    print("=" * 60)
    print("NeoGuard Dataset Downloader")
    print("=" * 60)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("\n[WARNING] No kaggle.json found at ~/.kaggle/kaggle.json")
        print("Download from: https://www.kaggle.com/settings → API → Create New Token")
        print("Then place the file at ~/.kaggle/kaggle.json")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownload directory: {DATA_DIR}\n")

    # Select which datasets to download
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
    else:
        selected = list(DATASETS.keys())

    for name in selected:
        if name not in DATASETS:
            print(f"  [SKIP] Unknown dataset: {name}")
            continue
        download_dataset(name, DATASETS[name])

    print("\nDone! Datasets are in:", DATA_DIR)


if __name__ == "__main__":
    main()
