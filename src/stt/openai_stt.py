"""
Speech-to-Text via AICredits (Whisper-compatible).
Uses POST /v1/audio/transcriptions
"""
from typing import Optional
from io import BytesIO
from loguru import logger
from src.aicredits_client import get_client


class OpenAISTT:
    def __init__(self, model: str = "openai/whisper-1"):
        self.client = get_client()
        self.model = model

    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """Transcribe audio bytes → text."""
        try:
            file_like = BytesIO(audio_bytes)
            file_like.name = filename
            transcript = self.client.audio.transcriptions.create(
                model=self.model,
                file=file_like,
                response_format="text",
            )
            text = transcript.strip() if isinstance(transcript, str) else str(transcript)
            logger.info(f"STT: {text}")
            return text
        except Exception as e:
            logger.error(f"STT error: {e}")
            # Fallback: try with a simpler model name if prefixed version fails
            try:
                file_like = BytesIO(audio_bytes)
                file_like.name = filename
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file_like,
                    response_format="text",
                )
                text = transcript.strip() if isinstance(transcript, str) else str(transcript)
                logger.info(f"STT (fallback): {text}")
                return text
            except Exception as e2:
                logger.error(f"STT fallback also failed: {e2}")
                return ""
