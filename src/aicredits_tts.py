"""
AICredits TTS for LiveKit Agents (non-streaming HTTP).
Calls /v1/audio/speech directly so Sarvam speaker names are not rewritten.
"""
from __future__ import annotations

import asyncio
import os
from io import BytesIO
from typing import Any

from loguru import logger

try:
    from livekit.agents import tts, APIConnectOptions
    from livekit import rtc
except ImportError:
    raise


SAMPLE_RATE = 24000
NUM_CHANNELS = 1


class AICreditsTTS(tts.TTS):
    """
    Direct AICredits TTS — supports:
      - openai/tts-1, openai/tts-1-hd  (voices: alloy, echo, fable, onyx, nova, shimmer)
      - sarvam/bulbul-v2               (voices: anushka, manisha, vidya, arya, abhilash, karun, hitesh)
      - sarvam/bulbul-v3               (voices: priya, ishita, simran, shubh, aditya, anand, ...)
    """

    def __init__(
        self,
        *,
        model: str = "sarvam/bulbul-v3",
        voice: str = "priya",
        base_url: str | None = None,
        api_key: str | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._model = model
        self._voice = voice
        self._base_url = (
            base_url or os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1"
        ).rstrip("/")
        self._api_key = api_key or os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("AICREDITS_API_KEY required")

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions | None = None,
    ) -> tts.ChunkedStream:
        return _ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options or APIConnectOptions(),
        )


class _ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: AICreditsTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        text = (self._input_text or "").strip()
        if not text:
            return

        audio_bytes = await asyncio.to_thread(self._synthesize_sync, text)
        if not audio_bytes:
            logger.error("AICredits TTS returned empty audio")
            return

        # Decode mp3 → PCM frames via pydub / ffmpeg
        try:
            from pydub import AudioSegment
            import numpy as np

            seg = AudioSegment.from_mp3(BytesIO(audio_bytes))
            seg = seg.set_frame_rate(self._tts.sample_rate).set_channels(1).set_sample_width(2)
            pcm = np.array(seg.get_array_of_samples(), dtype=np.int16)
        except Exception as e:
            logger.error(f"TTS audio decode failed (need pydub+ffmpeg): {e}")
            return

        request_id = "aicredits-tts"
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        # Push in ~20 ms chunks
        samples_per_frame = self._tts.sample_rate // 50
        for i in range(0, len(pcm), samples_per_frame):
            chunk = pcm[i : i + samples_per_frame]
            if len(chunk) == 0:
                continue
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self._tts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(chunk),
            )
            output_emitter.push(frame)

        output_emitter.flush()

    def _synthesize_sync(self, text: str) -> bytes:
        from openai import OpenAI

        client = OpenAI(base_url=self._tts._base_url, api_key=self._tts._api_key)
        models_to_try = [self._tts._model]
        # Fallbacks
        for m in ("sarvam/bulbul-v3", "sarvam/bulbul-v2", "tts-1", "openai/tts-1"):
            if m not in models_to_try:
                models_to_try.append(m)

        last_err = None
        for model in models_to_try:
            try:
                # Use native Sarvam speaker names when model is Sarvam
                voice = self._tts._voice
                logger.info(f"TTS request model={model} voice={voice} text={text[:60]!r}")
                resp = client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format="mp3",
                )
                data = resp.content
                if data:
                    logger.info(f"TTS ok model={model} voice={voice} → {len(data)} bytes")
                    return data
            except Exception as e:
                last_err = e
                logger.warning(f"TTS model={model} voice={self._tts._voice} failed: {e}")
        logger.error(f"All TTS models failed: {last_err}")
        return b""
