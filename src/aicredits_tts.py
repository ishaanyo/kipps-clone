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
            import numpy as np
            pcm = self._decode_to_pcm(audio_bytes, self._tts.sample_rate)
            if pcm is None or len(pcm) == 0:
                logger.error("TTS audio decode produced empty PCM")
                return
        except Exception as e:
            logger.error(f"TTS audio decode failed: {e}")
            return

        # LiveKit AudioEmitter expects PCM bytes (int16 little-endian)
        try:
            output_emitter.initialize(
                request_id="sarvam-tts",
                sample_rate=self._tts.sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
                stream=False,
            )
        except TypeError:
            # older SDK without stream kw
            output_emitter.initialize(
                request_id="sarvam-tts",
                sample_rate=self._tts.sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type="audio/pcm",
            )

        samples_per_frame = max(1, self._tts.sample_rate // 50)  # ~20ms
        pushed = 0
        pcm_bytes = np.ascontiguousarray(pcm, dtype=np.int16).tobytes()
        # Push whole buffer in small frames
        bytes_per_frame = samples_per_frame * NUM_CHANNELS * 2  # int16
        for i in range(0, len(pcm_bytes), bytes_per_frame):
            chunk = pcm_bytes[i : i + bytes_per_frame]
            if not chunk:
                continue
            # Prefer bytes push; fall back to AudioFrame
            try:
                output_emitter.push(chunk)
            except TypeError:
                n_samples = len(chunk) // (NUM_CHANNELS * 2)
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=self._tts.sample_rate,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=n_samples,
                )
                output_emitter.push(frame)
            pushed += 1

        try:
            output_emitter.flush()
        except Exception:
            pass
        logger.info(f"TTS pushed {pushed} frames ({len(pcm)} samples)")


    def _decode_to_pcm(self, audio_bytes: bytes, target_rate: int):
        """Decode wav/mp3 to int16 mono PCM; prefer paths that need no ffmpeg."""
        import numpy as np
        from io import BytesIO

        # 1) soundfile
        try:
            import soundfile as sf
            data, rate = sf.read(BytesIO(audio_bytes), dtype="int16", always_2d=True)
            mono = data.mean(axis=1).astype(np.int16) if data.shape[1] > 1 else data[:, 0]
            if rate != target_rate and len(mono) > 0:
                duration = len(mono) / float(rate)
                new_len = max(1, int(duration * target_rate))
                x_old = np.linspace(0, 1, num=len(mono), endpoint=False)
                x_new = np.linspace(0, 1, num=new_len, endpoint=False)
                mono = np.interp(x_new, x_old, mono.astype(np.float32)).astype(np.int16)
            logger.info(f"Decoded via soundfile: {len(mono)} samples @ {target_rate}Hz")
            return mono
        except Exception as e:
            logger.debug(f"soundfile decode: {e}")

        # 2) stdlib wave
        try:
            import wave
            with wave.open(BytesIO(audio_bytes), "rb") as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                sw = wf.getsampwidth()
                frames = wf.readframes(wf.getnframes())
            if sw == 2:
                pcm = np.frombuffer(frames, dtype=np.int16)
            elif sw == 1:
                pcm = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
            else:
                raise ValueError(f"unsupported sample width {sw}")
            if channels > 1:
                pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
            if rate != target_rate and len(pcm) > 0:
                duration = len(pcm) / float(rate)
                new_len = max(1, int(duration * target_rate))
                x_old = np.linspace(0, 1, num=len(pcm), endpoint=False)
                x_new = np.linspace(0, 1, num=new_len, endpoint=False)
                pcm = np.interp(x_new, x_old, pcm.astype(np.float32)).astype(np.int16)
            logger.info(f"Decoded via wave: {len(pcm)} samples @ {target_rate}Hz")
            return pcm
        except Exception as e:
            logger.debug(f"wave decode: {e}")

        # 3) pydub (needs ffmpeg for mp3)
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(BytesIO(audio_bytes))
            seg = seg.set_frame_rate(target_rate).set_channels(1).set_sample_width(2)
            pcm = np.array(seg.get_array_of_samples(), dtype=np.int16)
            logger.info(f"Decoded via pydub: {len(pcm)} samples")
            return pcm
        except Exception as e:
            logger.error(
                "All audio decoders failed. Install ffmpeg OR ensure Sarvam returns WAV. "
                f"Error: {e}"
            )
            return None

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
