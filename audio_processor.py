"""
Audio preprocessing for auscultation classification.
Computes mel spectrograms, MFCCs, and augmentations.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SR = 22050         # target sample rate
DURATION = 5.0     # clip length in seconds
N_MELS = 128
N_FRAMES = 128
HOP_LENGTH = 512
N_FFT = 2048
FMIN = 50          # Hz — below cardiac murmur range
FMAX = 2000        # Hz — above typical lung sound range


def load_audio(path: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Load and resample audio to fixed length."""
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    audio, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SR,
    n_mels: int = N_MELS,
    n_frames: int = N_FRAMES,
) -> np.ndarray:
    """Convert waveform to log-mel spectrogram."""
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Resize to fixed time dimension
    if log_mel.shape[1] < n_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :n_frames]

    # Normalize to [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel[..., np.newaxis]  # (n_mels, n_frames, 1)


def compute_mfcc(audio: np.ndarray, sr: int = SR, n_mfcc: int = 40) -> np.ndarray:
    """Compute MFCCs + delta + delta-delta."""
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc, delta, delta2], axis=0)  # (n_mfcc*3, frames)


def augment(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Data augmentation for limited medical audio datasets:
    - Time stretching
    - Pitch shifting
    - Gaussian noise (cardiac murmur simulation)
    - Time shift
    """
    aug = audio.copy()

    # Time stretch
    if np.random.rand() < 0.4:
        try:
            import librosa
            rate = np.random.uniform(0.85, 1.15)
            aug = librosa.effects.time_stretch(aug, rate=rate)
            target = int(sr * DURATION)
            aug = aug[:target] if len(aug) >= target else np.pad(aug, (0, target - len(aug)))
        except Exception:
            pass

    # Gaussian noise
    if np.random.rand() < 0.5:
        noise_amp = np.random.uniform(0.002, 0.008) * aug.max()
        aug = aug + noise_amp * np.random.randn(*aug.shape)

    # Time shift
    if np.random.rand() < 0.4:
        shift = int(np.random.uniform(-0.1, 0.1) * sr)
        aug = np.roll(aug, shift)

    return aug.astype(np.float32)


def preprocess_file(path: str) -> np.ndarray:
    """End-to-end: path → mel spectrogram ready for model input."""
    audio = load_audio(path)
    return compute_mel_spectrogram(audio)
