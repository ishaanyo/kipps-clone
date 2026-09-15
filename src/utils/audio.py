import numpy as np
import sounddevice as sd
import soundfile as sf
from io import BytesIO
from typing import Optional
import wave

SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(duration: float = 5.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from microphone."""
    print(f"🎤 Recording for {duration}s... Speak now!")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Play audio through speakers."""
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE):
    """Save numpy audio to WAV file."""
    sf.write(path, audio, sample_rate)


def bytes_to_numpy(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convert raw PCM bytes to numpy array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy array to WAV bytes."""
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
