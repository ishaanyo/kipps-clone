import os
"""
Text-to-Speech via AICredits.
Uses POST /v1/audio/speech
Supported models: openai/tts-1, openai/tts-1-hd, sarvam/bulbul-v2
"""
from typing import Optional
from loguru import logger
from src.aicredits_client import get_client
import numpy as np
from io import BytesIO


class OpenAITTS:
    def __init__(
        self,
        model: str = "openai/tts-1",
        voice: str = "alloy",
    ):
        self.client = get_client()
        self.model = model
        self.voice = voice  # alloy, echo, fable, onyx, nova, shimmer

    def synthesize(self, text: str) -> bytes:
        """Generate audio bytes (mp3)."""
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                response_format="mp3",
            )
            audio_bytes = response.content
            logger.info(f"TTS generated {len(audio_bytes)} bytes for: {text[:50]}...")
            return audio_bytes
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Retry with short model name
            try:
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=self.voice,
                    input=text,
                    response_format="mp3",
                )
                return response.content
            except Exception as e2:
                logger.error(f"TTS fallback failed: {e2}")
                return b""

    def synthesize_to_numpy(self, text: str) -> np.ndarray:
        """Return numpy array suitable for playback."""
        audio_bytes = self.synthesize(text)
        if not audio_bytes:
            return np.array([])
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(BytesIO(audio_bytes))
            audio = audio.set_frame_rate(16000).set_channels(1)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            return samples
        except Exception as e:
            logger.warning(f"Audio conversion failed (install pydub + ffmpeg): {e}")
            return np.array([])

    def synthesize_to_file(self, text: str, suffix: str = ".mp3") -> str:
        """Write TTS audio to a temp file and return path (for Gradio playback)."""
        import tempfile
        audio_bytes = self.synthesize(text)
        if not audio_bytes:
            return ""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            os.write(fd, audio_bytes)
        finally:
            os.close(fd)
        return path
