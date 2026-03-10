"""
Audio Feature Extractor — NeoGuard Inference Pipeline
======================================================
Extracts 33 features from 1-second audio chunks:

  13 MFCC means      — average frequency pattern of the cry
  13 MFCC stds       — how much those frequencies varied
   1 Spectral centroid — center of mass of the sound
   1 Spectral bandwidth — spread of frequencies
   1 Spectral rolloff  — where high frequencies drop off
   1 ZCR               — zero crossing rate (roughness)
   1 RMS               — loudness / energy
   1 F0 mean           — average pitch
   1 F0 std            — pitch variation
  ─────────────────────────────
  33 total features
"""

import numpy as np
import librosa
from typing import Optional


class AudioFeatureExtractor:
    """
    Extracts 33 cry-detection features from audio.

    Usage:
        extractor = AudioFeatureExtractor()
        features = extractor.extract_from_file("cry_clip.wav")
        # features: np.ndarray of shape (33,)
    """

    N_MFCC = 13
    SR = 22050       # target sample rate
    N_FFT = 512
    HOP_LENGTH = 256

    def extract(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract 33 features from audio signal.

        Args:
            y:  Audio time series (mono)
            sr: Sample rate

        Returns:
            np.ndarray of shape (33,)
        """
        # Resample to standard rate if needed
        if sr != self.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)
            sr = self.SR

        # 1. MFCC (13 means + 13 stds = 26 features)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
        )
        mfcc_means = np.mean(mfcc, axis=1)   # (13,)
        mfcc_stds  = np.std(mfcc, axis=1)    # (13,)

        # 2. Spectral centroid — 1 feature
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
            )
        ))

        # 3. Spectral bandwidth — 1 feature
        spectral_bandwidth = float(np.mean(
            librosa.feature.spectral_bandwidth(
                y=y, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
            )
        ))

        # 4. Spectral rolloff — 1 feature
        spectral_rolloff = float(np.mean(
            librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
            )
        ))

        # 5. Zero crossing rate — 1 feature
        zcr = float(np.mean(
            librosa.feature.zero_crossing_rate(y, hop_length=self.HOP_LENGTH)
        ))

        # 6. RMS energy — 1 feature
        rms = float(np.mean(
            librosa.feature.rms(y=y, hop_length=self.HOP_LENGTH)
        ))

        # 7. F0 (fundamental frequency / pitch) — mean + std = 2 features
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_valid = f0[~np.isnan(f0)] if f0 is not None else np.array([0.0])
        f0_mean = float(np.mean(f0_valid)) if len(f0_valid) > 0 else 0.0
        f0_std  = float(np.std(f0_valid))  if len(f0_valid) > 0 else 0.0

        # Concatenate all 33 features
        features = np.concatenate([
            mfcc_means,           # 13
            mfcc_stds,            # 13
            [spectral_centroid],  #  1
            [spectral_bandwidth], #  1
            [spectral_rolloff],   #  1
            [zcr],                #  1
            [rms],                #  1
            [f0_mean],            #  1
            [f0_std],             #  1
        ]).astype(np.float32)     # = 33

        return features

    def extract_from_file(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract 33 features from an audio file (wav, mp3, etc.)."""
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            return self.extract(y, sr)
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None

    def extract_from_bytes(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Extract 33 features from raw audio bytes."""
        import io
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
            return self.extract(y, sr)
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None
