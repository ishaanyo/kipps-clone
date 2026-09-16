"""
TTS for LiveKit Agents via AICredits or native Sarvam API.

Priority:
1. If SARVAM_API_KEY is set → call Sarvam API directly (correct speakers)
2. Else → call AICredits /v1/audio/speech
"""
from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO

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
    Direct TTS — supports:
      Native Sarvam (SARVAM_API_KEY):
        model: bulbul:v3 / bulbul:v2
        speaker: priya, ishita, anushka, shubh, ...
      AICredits:
        model: sarvam/bulbul-v3, sarvam/bulbul-v2, tts-1
        voice: same names or alloy/nova/...
    """

    def __init__(
        self,
        *,
        model: str = "sarvam/bulbul-v3",
        voice: str = "priya",
        base_url: str | None = None,
        api_key: str | None = None,
        sarvam_api_key: str | None = None,
        language_code: str = "hi-IN",
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._model = model
        self._voice = voice
        self._language_code = language_code
        self._base_url = (
            base_url or os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1"
        ).rstrip("/")
        self._api_key = api_key or os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._sarvam_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
        if not self._api_key and not self._sarvam_key:
            raise ValueError("AICREDITS_API_KEY or SARVAM_API_KEY required")

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
            logger.error("TTS returned empty audio")
            return

        try:
            from pydub import AudioSegment
            import numpy as np

            bio = BytesIO(audio_bytes)
            try:
                seg = AudioSegment.from_file(bio)
            except Exception:
                bio.seek(0)
                seg = AudioSegment.from_mp3(bio)
            seg = (
                seg.set_frame_rate(self._tts.sample_rate)
                .set_channels(1)
                .set_sample_width(2)
            )
            pcm = np.array(seg.get_array_of_samples(), dtype=np.int16)
        except Exception as e:
            logger.error(f"TTS audio decode failed (need pydub+ffmpeg): {e}")
            return

        output_emitter.initialize(
            request_id="sarvam-tts",
            sample_rate=self._tts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

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
        if self._tts._sarvam_key:
            data = self._sarvam_tts(text)
            if data:
                return data

        if self._tts._api_key:
            data = self._aicredits_tts(text)
            if data:
                return data

        return b""

    def _normalize_sarvam_model(self, model: str) -> str:
        m = (model or "").lower().strip()
        if "v3" in m:
            return "bulbul:v3"
        if "v2" in m or "bulbul" in m:
            return "bulbul:v2"
        return "bulbul:v3"

    def _sarvam_tts(self, text: str) -> bytes:
        """Call Sarvam REST TTS directly — correct speaker names."""
        import httpx

        model = self._normalize_sarvam_model(self._tts._model)
        speaker = (self._tts._voice or "priya").lower().strip()
        lang = self._tts._language_code or "hi-IN"

        url = "https://api.sarvam.ai/text-to-speech"
        headers = {
            "api-subscription-key": self._tts._sarvam_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "target_language_code": lang,
            "model": model,
            "speaker": speaker,
            "speech_sample_rate": self._tts.sample_rate,
            "enable_preprocessing": True,
        }
        logger.info(f"Sarvam TTS model={model} speaker={speaker} lang={lang}")
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(url, headers=headers, json=payload)
                if r.status_code != 200:
                    logger.warning(f"Sarvam TTS HTTP {r.status_code}: {r.text[:300]}")
                    return b""
                data = r.json()
                audios = data.get("audios") or data.get("audio") or []
                if isinstance(audios, str):
                    audios = [audios]
                if not audios:
                    logger.warning(f"Sarvam TTS empty response keys={list(data.keys())}")
                    return b""
                raw = base64.b64decode(audios[0])
                logger.info(f"Sarvam TTS ok → {len(raw)} bytes")
                return raw
        except Exception as e:
            logger.error(f"Sarvam TTS failed: {e}")
            return b""

    def _aicredits_tts(self, text: str) -> bytes:
        from openai import OpenAI

        client = OpenAI(base_url=self._tts._base_url, api_key=self._tts._api_key)
        models_to_try = [self._tts._model]
        for m in ("sarvam/bulbul-v3", "sarvam/bulbul-v2", "tts-1", "openai/tts-1"):
            if m not in models_to_try:
                models_to_try.append(m)

        for model in models_to_try:
            try:
                voice = self._tts._voice
                logger.info(f"AICredits TTS model={model} voice={voice}")
                resp = client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format="mp3",
                )
                data = resp.content
                if data:
                    logger.info(f"AICredits TTS ok → {len(data)} bytes")
                    return data
            except Exception as e:
                logger.warning(f"AICredits TTS model={model} failed: {e}")
        return b""
