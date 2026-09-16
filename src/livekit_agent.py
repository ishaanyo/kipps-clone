"""
LiveKit voice agent — AICredits STT/LLM/TTS.
Hinglish-friendly models (Sarvam when available).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

if os.getenv("AICREDITS_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("AICREDITS_API_KEY")

from loguru import logger

try:
    from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
    from livekit.plugins import openai, silero
except ImportError:
    print('Install: pip install "livekit-agents[openai,silero]" livekit-api livekit')
    raise SystemExit(1)

from src.utils.lead_store import LeadStore

AICREDITS_BASE = (os.getenv("AICREDITS_BASE_URL") or "https://api.aicredits.in/v1").rstrip("/")
AICREDITS_KEY = os.getenv("AICREDITS_API_KEY") or os.getenv("OPENAI_API_KEY")

# Multilingual / India-friendly defaults (override in .env)
# STT: Sarvam Saarika is strong for Indian languages; Whisper as fallback
STT_MODEL = os.getenv("LIVEKIT_STT_MODEL", "sarvam/saarika-v2")
STT_FALLBACK = os.getenv("LIVEKIT_STT_FALLBACK", "whisper-1")
LLM_MODEL = os.getenv("LIVEKIT_LLM_MODEL", "gpt-4o-mini")  # solid Hindi/Hinglish

# TTS – native Sarvam when SARVAM_API_KEY is set (correct speakers)
TTS_MODEL = os.getenv("LIVEKIT_TTS_MODEL", "sarvam/bulbul-v3")
TTS_FALLBACK = os.getenv("LIVEKIT_TTS_FALLBACK", "tts-1")
TTS_VOICE = os.getenv("LIVEKIT_TTS_VOICE", "priya")  # priya, ishita, anushka, shubh, aditya...
TTS_LANG = os.getenv("LIVEKIT_TTS_LANGUAGE", "hi-IN")

INSTRUCTIONS = """
You are SecureLoan Finance's live phone AI agent for India.

LANGUAGE (must follow):
- Always reply in the same language mix the caller is using RIGHT NOW.
- Hindi only → pure simple Hindi (spoken style).
- Hinglish → natural Hinglish (e.g. "Bilkul sir, home loan ke liye kitna amount chahiye?").
- English only → simple clear English.
- Never force English if the user spoke Hindi/Hinglish.
- Keep answers short for speech (1-3 sentences). No markdown or symbols.

PRODUCT:
Departments: Personal Loan, Home Loan, Business Loan, Gold Loan, Credit Card, Customer Support, Collections.
Ask department if unclear. Qualify amount, city, income when relevant. Offer callback with name + phone.
Never invent rates. Always respond to the last user message.
"""


def _make_stt():
    """Prefer official Sarvam plugin (multi-turn stable); else AICredits STT."""
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if sarvam_key:
        try:
            from livekit.plugins import sarvam
            # Non-realtime STT is widely available; good for multi-turn
            try:
                stt = sarvam.STT(
                    language="hi-IN",
                    model="saaras:v4",
                    mode="codemix",
                    high_vad_sensitivity=True,
                )
            except TypeError:
                stt = sarvam.STT(language="hi-IN")
            logger.info("STT = official Sarvam plugin (saaras)")
            return stt
        except Exception as e:
            logger.error(f"Official Sarvam STT unavailable — run: pip install livekit-plugins-sarvam | {e}")

    from src.aicredits_stt import AICreditsSTT
    stt = AICreditsSTT(
        model=STT_MODEL,
        language=None,
        base_url=AICREDITS_BASE,
        api_key=AICREDITS_KEY,
        fallback_model=STT_FALLBACK,
    )
    logger.info(f"STT primary={STT_MODEL} fallback={STT_FALLBACK} (AICredits)")
    return stt


def _make_llm():
    kwargs = dict(model=LLM_MODEL, api_key=AICREDITS_KEY, temperature=0.55)
    try:
        return openai.LLM(**kwargs, base_url=AICREDITS_BASE)
    except TypeError:
        return openai.LLM(**kwargs)


def _make_tts():
    """Prefer official Sarvam TTS plugin; else custom AICredits TTS."""
    sarvam_key = os.getenv("SARVAM_API_KEY")
    voice = (TTS_VOICE or "priya").strip().lower()
    lang = (TTS_LANG or "hi-IN").strip()

    if sarvam_key:
        try:
            from livekit.plugins import sarvam
            model = "bulbul:v3" if "v3" in (TTS_MODEL or "").lower() else "bulbul:v2"
            try:
                tts = sarvam.TTS(
                    target_language_code=lang,
                    model=model,
                    speaker=voice,
                    speech_sample_rate=22050,
                    pace=1.0,
                )
            except TypeError:
                tts = sarvam.TTS(
                    target_language_code=lang,
                    model=model,
                    speaker=voice,
                )
            logger.info(f"TTS = official Sarvam plugin model={model} speaker={voice} lang={lang}")
            return tts
        except Exception as e:
            logger.error(f"Official Sarvam TTS unavailable — run: pip install livekit-plugins-sarvam | {e}")

    from src.aicredits_tts import AICreditsTTS
    tts = AICreditsTTS(
        model=TTS_MODEL or "sarvam/bulbul-v3",
        voice=voice,
        base_url=AICREDITS_BASE,
        api_key=AICREDITS_KEY,
        sarvam_api_key=sarvam_key,
        language_code=lang,
    )
    logger.info(f"TTS model={TTS_MODEL} voice={voice} lang={lang} (custom fallback)")
    return tts


def _build_session() -> AgentSession:
    if not AICREDITS_KEY and not os.getenv("SARVAM_API_KEY"):
        raise RuntimeError("AICREDITS_API_KEY or SARVAM_API_KEY required")
    vad = silero.VAD.load()
    kwargs = dict(stt=_make_stt(), llm=_make_llm(), tts=_make_tts(), vad=vad)
    try:
        return AgentSession(**kwargs, allow_interruptions=True, min_endpointing_delay=0.8)
    except TypeError:
        return AgentSession(**kwargs)


async def entrypoint(ctx: JobContext):
    # PLUGIN CHECK
    if os.getenv("SARVAM_API_KEY"):
        try:
            from livekit.plugins import sarvam  # noqa: F401
            logger.info("livekit.plugins.sarvam is installed")
        except Exception as e:
            logger.error(
                "SARVAM_API_KEY set but livekit-plugins-sarvam NOT installed. "
                "Run: pip install livekit-plugins-sarvam  | %s", e
            )

    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    session = _build_session()
    agent = Agent(instructions=INSTRUCTIONS)
    lead_store = LeadStore()
    transcript_parts: list[dict] = []
    saved = {"done": False}

    def save_lead(reason: str = ""):
        if saved["done"]:
            return
        saved["done"] = True
        try:
            msgs = list(transcript_parts) or [
                {"role": "system", "content": f"Call ended ({reason})"}
            ]
            path = lead_store.save_conversation(
                messages=msgs,
                department="livekit_call",
                extra={"room": ctx.room.name, "reason": reason},
            )
            logger.success(f"Lead saved ({reason}) → {path}")
        except Exception as e:
            logger.error(f"Lead save failed: {e}")

    @session.on("user_input_transcribed")
    def on_user(ev):
        text = getattr(ev, "transcript", None) or ""
        is_final = getattr(ev, "is_final", True)
        text = str(text).strip()
        # Ignore empty / JSON noise from STT
        if text.startswith("{") and "transcript" in text:
            try:
                import json
                obj = json.loads(text)
                text = (obj.get("transcript") or obj.get("text") or "").strip()
            except Exception:
                text = ""
        # Ignore filler / echo (agent hearing itself)
        low = text.lower().strip(" .,!?;:")
        fillers = {
            "hmm", "hm", "hmmm", "hmm hmm", "hmm hmm hmm",
            "uh", "um", "ah", "aa", "haan", "ha", "ok", "okay",
            "mm", "mmm", "hmm hmm hmm hmm",
        }
        if low in fillers or set(low.replace(" ", "")) <= set("hm"):
            logger.info(f"USER STT ignored filler: {text!r}")
            return
        logger.info(f"USER STT final={is_final}: {text!r}")
        if is_final and text:
            transcript_parts.append({"role": "user", "content": text})

    @session.on("conversation_item_added")
    def on_item(ev):
        try:
            item = ev.item
            role = getattr(item, "role", None)
            content = getattr(item, "text_content", None) or getattr(item, "content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            content = str(content or "").strip()
            if role and content:
                logger.info(f"ITEM {role}: {content[:120]!r}")
                transcript_parts.append({"role": str(role), "content": content})
        except Exception as e:
            logger.warning(f"item: {e}")

    @session.on("close")
    def on_close(ev=None):
        save_lead("session_close")

    await session.start(room=ctx.room, agent=agent)

    await session.generate_reply(
        instructions=(
            "Caller se short friendly greeting Hinglish mein do — "
            "jaise real call center. SecureLoan Finance welcome, "
            "poocho kaunsa department chahiye. Sirf 2 short sentences."
        )
    )

    @ctx.room.on("participant_disconnected")
    def on_disconnect(participant):
        logger.info(f"Participant left: {participant.identity}")
        save_lead("participant_disconnected")
        # Shut down agent session so worker releases the job
        try:
            import asyncio
            asyncio.create_task(session.aclose())
        except Exception as e:
            logger.warning(f"session.aclose: {e}")
            try:
                asyncio.create_task(ctx.room.disconnect())
            except Exception:
                pass

    @ctx.room.on("disconnected")
    def on_room_disconnected():
        save_lead("room_disconnected")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
