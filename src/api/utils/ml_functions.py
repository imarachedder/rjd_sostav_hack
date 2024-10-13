import librosa
import numpy as np


def process_audio(audio_bytes: bytes, sr: int = 22050, duration: float = 2.5) -> np.ndarray:
    """
    Processes raw audio bytes and extracts features.

    Args:
        audio_bytes (bytes): Raw audio data.
        sr (int): Sampling rate.
        duration (float): Duration to which the audio is trimmed or padded.

    Returns:
        np.ndarray: Extracted features.
    """
    try:
        # Load audio from bytes
        y, sr = librosa.load(librosa.util.buf_to_float(audio_bytes, n_bytes=2), sr=sr, duration=duration)

        # Example feature: Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Compute mean of MFCCs for simplicity
        mfcc_mean = np.mean(mfcc, axis=1)

        return mfcc_mean
    except Exception as e:
        raise ValueError(f"Error processing audio: {e}")


def predict_label(model, features: np.ndarray) -> str:
    """
    Predicts the label using the ML model.

    Args:
        model: Trained ML model.
        features (np.ndarray): Extracted audio features.

    Returns:
        str: Predicted label.
    """
    try:
        # Reshape features for prediction
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return prediction
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")