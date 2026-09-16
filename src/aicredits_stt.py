"""
AICredits Whisper STT for LiveKit Agents (non-streaming HTTP).
"""
from __future__ import annotations

import asyncio
import os
from io import BytesIO
from typing import Any

from loguru import logger

try:
    from livekit.agents import stt, APIConnectOptions
    from livekit import rtc
except ImportError:
    raise


def _wav_from_buffer(buffer: Any) -> bytes:
    """Convert LiveKit audio buffer to WAV bytes across SDK versions."""
    try:
        if hasattr(rtc, "combine_audio_frames"):
            frame = rtc.combine_audio_frames(buffer)
            if hasattr(frame, "to_wav_bytes"):
                return frame.to_wav_bytes()
    except Exception as e:
        logger.debug(f"combine_audio_frames: {e}")

    # buffer may already be list of frames
    frames = buffer if isinstance(buffer, (list, tuple)) else [buffer]
    try:
        from livekit.agents.utils import audio as audio_utils
        if hasattr(audio_utils, "merge_frames"):
            merged = audio_utils.merge_frames(frames)
            if hasattr(merged, "to_wav_bytes"):
                return merged.to_wav_bytes()
    except Exception:
        pass

    # Last resort: use first frame raw + minimal wav header via soundfile
    import numpy as np
    import soundfile as sf
    samples = []
    sample_rate = 16000
    for f in frames:
        sample_rate = getattr(f, "sample_rate", sample_rate) or 16000
        data = getattr(f, "data", None)
        if data is not None:
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            samples.append(arr)
    if not samples:
        return b""
    audio = np.concatenate(samples)
    bio = BytesIO()
    sf.write(bio, audio, sample_rate, format="WAV")
    return bio.getvalue()


class AICreditsSTT(stt.STT):
    def __init__(
        self,
        *,
        model: str = "whisper-1",
        language: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        fallback_model: str | None = "whisper-1",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = model
        self._fallback_model = fallback_model
        self._language = language
        self._base_url = (
            base_url or os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1"
        ).rstrip("/")
        self._api_key = api_key or os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("AICREDITS_API_KEY required")

    async def _recognize_impl(
        self,
        buffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions | None = None,
    ) -> stt.SpeechEvent:
        lang = language or self._language
        wav = _wav_from_buffer(buffer)
        if not wav:
            logger.warning("STT: empty audio buffer")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(text="", language=lang or "en")],
            )

        logger.info(f"STT: {len(wav)} bytes → AICredits")
        text = await asyncio.to_thread(self._transcribe_sync, wav, lang)
        logger.info(f"STT text: {text!r}")

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text or "", language=lang or "en")],
        )

    def _transcribe_sync(self, wav: bytes, lang: str | None) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        f = BytesIO(wav)
        f.name = "audio.wav"
        kwargs = {"model": self._model, "file": f, "response_format": "text"}
        if lang:
            kwargs["language"] = lang
        models_to_try = []
        for mid in (self._model, self._fallback_model, "whisper-1", "openai/whisper-1"):
            if mid and mid not in models_to_try:
                models_to_try.append(mid)

        last_err = None
        for mid in models_to_try:
            try:
                f = BytesIO(wav)
                f.name = "audio.wav"
                kw = {"model": mid, "file": f, "response_format": "text"}
                # Prefer Hindi/English mix for India calls
                if lang:
                    kw["language"] = lang
                elif "sarvam" not in (mid or "").lower():
                    kw["language"] = "hi"  # help Whisper on Hinglish
                out = client.audio.transcriptions.create(**kw)
                text_out = out if isinstance(out, str) else (getattr(out, "text", None) or str(out))
                text_out = (text_out or "").strip()
                # Sarvam often returns JSON: {"transcript":"...", "language_code":"hi-IN"}
                if text_out.startswith("{") and ("transcript" in text_out or "text" in text_out):
                    try:
                        import json
                        obj = json.loads(text_out)
                        parsed = obj.get("transcript")
                        if parsed is None:
                            parsed = obj.get("text")
                        text_out = (parsed if parsed is not None else "").strip()
                    except Exception:
                        text_out = ""
                if text_out:
                    low = text_out.lower().strip(" .,!?;:")
                    if low in {"hmm","hm","hmmm","hmm hmm","hmm hmm hmm","uh","um","mm","mmm"} or set(low.replace(" ","")) <= set("hm"):
                        logger.info(f"STT filler ignored model={mid} → {text_out!r}")
                        return ""
                    logger.info(f"STT ok model={mid} → {text_out!r}")
                    return text_out
                logger.info(f"STT empty transcript model={mid} — trying next")
            except Exception as e:
                last_err = e
                logger.warning(f"STT model={mid} failed: {e}")
        logger.error(f"All STT models failed or empty: {last_err}")
        return ""
